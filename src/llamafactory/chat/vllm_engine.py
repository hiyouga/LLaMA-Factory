# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Sequence, Union

from typing_extensions import override

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import IMAGE_PLACEHOLDER
from ..extras.misc import get_device_count
from ..extras.packages import is_pillow_available, is_vllm_available
from ..model import load_config, load_tokenizer
from ..model.model_utils.quantization import QuantizationMethod
from ..model.model_utils.visual import LlavaMultiModalProjectorForYiVLForVLLM
from .base_engine import BaseEngine, Response


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_vllm_available():
    from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
    from vllm.lora.request import LoRARequest


if TYPE_CHECKING:
    from ..data.mm_plugin import ImageInput, VideoInput
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class VllmEngine(BaseEngine):
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        config = load_config(model_args)  # may download model from ms hub
        if getattr(config, "quantization_config", None):  # gptq models should use float16
            quantization_config: Dict[str, Any] = getattr(config, "quantization_config", None)
            quant_method = quantization_config.get("quant_method", "")
            if quant_method == QuantizationMethod.GPTQ and model_args.infer_dtype == "auto":
                model_args.infer_dtype = "float16"

        self.can_generate = finetuning_args.stage == "sft"
        tokenizer_module = load_tokenizer(model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        self.tokenizer.padding_side = "left"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        self.generating_args = generating_args.to_dict()

        engine_args = {
            "model": model_args.model_name_or_path,
            "trust_remote_code": True,
            "download_dir": model_args.cache_dir,
            "dtype": model_args.infer_dtype,
            "max_model_len": model_args.vllm_maxlen,
            "tensor_parallel_size": get_device_count() or 1,
            "gpu_memory_utilization": model_args.vllm_gpu_util,
            "disable_log_stats": True,
            "disable_log_requests": True,
            "enforce_eager": model_args.vllm_enforce_eager,
            "enable_lora": model_args.adapter_name_or_path is not None,
            "max_lora_rank": model_args.vllm_max_lora_rank,
        }

        if getattr(config, "is_yi_vl_derived_model", None):
            import vllm.model_executor.models.llava

            logger.info_rank0("Detected Yi-VL model, applying projector patch.")
            vllm.model_executor.models.llava.LlavaMultiModalProjector = LlavaMultiModalProjectorForYiVLForVLLM

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
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        **input_kwargs,
    ) -> AsyncIterator["RequestOutput"]:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        if images is not None:
            if not any(IMAGE_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = IMAGE_PLACEHOLDER * len(images) + messages[0]["content"]

        paired_messages = messages + [{"role": "assistant", "content": ""}]
        system = system or self.generating_args["default_system"]
        prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, paired_messages, system, tools)
        prompt_length = len(prompt_ids)

        temperature: Optional[float] = input_kwargs.pop("temperature", None)
        top_p: Optional[float] = input_kwargs.pop("top_p", None)
        top_k: Optional[float] = input_kwargs.pop("top_k", None)
        num_return_sequences: int = input_kwargs.pop("num_return_sequences", 1)
        repetition_penalty: Optional[float] = input_kwargs.pop("repetition_penalty", None)
        length_penalty: Optional[float] = input_kwargs.pop("length_penalty", None)
        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        max_new_tokens: Optional[int] = input_kwargs.pop("max_new_tokens", None)
        stop: Optional[Union[str, List[str]]] = input_kwargs.pop("stop", None)

        if length_penalty is not None:
            logger.warning_rank0("Length penalty is not supported by the vllm engine yet.")

        if "max_new_tokens" in self.generating_args:
            max_tokens = self.generating_args["max_new_tokens"]
        elif "max_length" in self.generating_args:
            if self.generating_args["max_length"] > prompt_length:
                max_tokens = self.generating_args["max_length"] - prompt_length
            else:
                max_tokens = 1

        if max_length:
            max_tokens = max_length - prompt_length if max_length > prompt_length else 1

        if max_new_tokens:
            max_tokens = max_new_tokens

        sampling_params = SamplingParams(
            n=num_return_sequences,
            repetition_penalty=(
                repetition_penalty if repetition_penalty is not None else self.generating_args["repetition_penalty"]
            )
            or 1.0,  # repetition_penalty must > 0
            temperature=temperature if temperature is not None else self.generating_args["temperature"],
            top_p=(top_p if top_p is not None else self.generating_args["top_p"]) or 1.0,  # top_p must > 0
            top_k=top_k if top_k is not None else self.generating_args["top_k"],
            stop=stop,
            stop_token_ids=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )

        if images is not None:  # add image features
            image_data = []
            for image in images:
                if not isinstance(image, (str, ImageObject)):
                    raise ValueError(f"Expected image input is a path or PIL.Image, but got {type(image)}.")

                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")

                image_data.append(image)

            multi_modal_data = {"image": image_data}
        else:
            multi_modal_data = None

        result_generator = self.model.generate(
            inputs={"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data},
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=self.lora_request,
        )
        return result_generator

    @override
    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        **input_kwargs,
    ) -> List["Response"]:
        final_output = None
        generator = await self._generate(messages, system, tools, images, videos, **input_kwargs)
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

    @override
    async def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        generated_text = ""
        generator = await self._generate(messages, system, tools, images, videos, **input_kwargs)
        async for result in generator:
            delta_text = result.outputs[0].text[len(generated_text) :]
            generated_text = result.outputs[0].text
            yield delta_text

    @override
    async def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        raise NotImplementedError("vLLM engine does not support get_scores.")
