import uuid
from typing import TYPE_CHECKING, AsyncGenerator, AsyncIterator, Dict, List, Optional, Sequence

from ..data import get_template_and_fix_tokenizer
from ..extras.misc import get_device_count, infer_optim_dtype
from ..extras.packages import is_vllm_available
from ..model import load_config, load_tokenizer
from .base_engine import BaseEngine, Response


if is_vllm_available():
    from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sequence import MultiModalData


if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray
    from transformers.image_processing_utils import BaseImageProcessor

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


class VllmEngine(BaseEngine):
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        config = load_config(model_args)  # may download model from ms hub
        infer_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))
        infer_dtype = str(infer_dtype).split(".")[-1]

        self.can_generate = finetuning_args.stage == "sft"
        tokenizer_module = load_tokenizer(model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        self.tokenizer.padding_side = "left"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args.template)
        self.generating_args = generating_args.to_dict()

        engine_args = {
            "model": model_args.model_name_or_path,
            "trust_remote_code": True,
            "download_dir": model_args.cache_dir,
            "dtype": infer_dtype,
            "max_model_len": model_args.vllm_maxlen,
            "tensor_parallel_size": get_device_count() or 1,
            "gpu_memory_utilization": model_args.vllm_gpu_util,
            "disable_log_stats": True,
            "disable_log_requests": True,
            "enforce_eager": model_args.vllm_enforce_eager,
            "enable_lora": model_args.adapter_name_or_path is not None,
        }

        if model_args.visual_inputs:
            # TODO: auto derive from config
            # https://github.com/vllm-project/vllm/pull/3042#issuecomment-1984893549
            self.image_feature_size = 576
            engine_args["image_input_type"] = "pixel_values"
            engine_args["image_token_id"] = self.tokenizer.convert_tokens_to_ids("<image>")
            engine_args["image_input_shape"] = "1,3,336,336"
            engine_args["image_feature_size"] = self.image_feature_size

        self.model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
        if model_args.adapter_name_or_path is not None:
            self.lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
        else:
            self.lora_request = None

    async def _generate(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> AsyncIterator["RequestOutput"]:
        request_id = "chatcmpl-{}".format(uuid.uuid4().hex)
        if self.processor is not None and image is not None and "<image>" not in messages[0]["content"]:
            messages[0]["content"] = "<image>" * self.image_feature_size + messages[0]["content"]

        paired_messages = messages + [{"role": "assistant", "content": ""}]
        prompt_ids, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer, messages=paired_messages, system=system, tools=tools
        )
        prompt_length = len(prompt_ids)

        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        num_return_sequences = input_kwargs.pop("num_return_sequences", None)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)

        generating_args = self.generating_args.copy()
        generating_args.update(
            dict(
                temperature=temperature or generating_args["temperature"],
                top_p=top_p or generating_args["top_p"],
                top_k=top_k or generating_args["top_k"],
                num_return_sequences=num_return_sequences or 1,
                repetition_penalty=repetition_penalty or generating_args["repetition_penalty"],
            )
        )

        if max_length:
            generating_args["max_new_tokens"] = max_length - prompt_length

        if max_new_tokens:
            generating_args["max_new_tokens"] = max_new_tokens

        sampling_params = SamplingParams(
            n=generating_args["num_return_sequences"],
            repetition_penalty=generating_args["repetition_penalty"],
            temperature=generating_args["temperature"],
            top_p=generating_args["top_p"],
            top_k=generating_args["top_k"],
            use_beam_search=generating_args["num_beams"] > 1,
            length_penalty=generating_args["length_penalty"],
            stop_token_ids=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            max_tokens=generating_args["max_new_tokens"],
            skip_special_tokens=True,
        )

        if self.processor is not None and image is not None:
            image_processor: "BaseImageProcessor" = getattr(self.processor, "image_processor")
            pixel_values: "torch.Tensor" = image_processor(image, return_tensors="pt")["pixel_values"]
            multi_modal_data = MultiModalData(type=MultiModalData.Type.IMAGE, data=pixel_values)
        else:
            multi_modal_data = None

        result_generator = self.model.generate(
            prompt=None,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_token_ids=prompt_ids,
            lora_request=self.lora_request,
            multi_modal_data=multi_modal_data,
        )
        return result_generator

    async def start(self) -> None:
        pass

    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> List["Response"]:
        final_output = None
        generator = await self._generate(messages, system, tools, image, **input_kwargs)
        async for request_output in generator:
            final_output = request_output

        results = []
        for output in final_output.outputs:
            results.append(
                Response(
                    response_text=output.text,
                    response_length=len(output.token_ids),
                    prompt_length=len(final_output.prompt_token_ids),
                    finish_reason=output.finish_reason,
                )
            )

        return results

    async def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        generated_text = ""
        generator = await self._generate(messages, system, tools, image, **input_kwargs)
        async for result in generator:
            delta_text = result.outputs[0].text[len(generated_text) :]
            generated_text = result.outputs[0].text
            yield delta_text

    async def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        raise NotImplementedError("vLLM engine does not support get_scores.")
