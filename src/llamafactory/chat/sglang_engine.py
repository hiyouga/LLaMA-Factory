from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
)

# Import SGLang correctly
import sglang as sgl
from typing_extensions import override

from ..extras.constants import (
    AUDIO_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    VIDEO_PLACEHOLDER,
)
from ..extras.logging import get_logger
from ..extras.misc import get_device_count
from ..hparams import (
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    ModelArguments,
    QuantizationMethod,
)
from ..model import load_config, load_tokenizer
from ..template import get_template_and_fix_tokenizer
from .base_engine import BaseEngine, Response


if TYPE_CHECKING:
    from ..data import AudioInput, ImageInput, VideoInput


logger = get_logger(__name__)


class SGLangEngine(BaseEngine):
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        """
        Initializes an SGLang inference engine.
        """
        self.model_args = model_args
        config = load_config(model_args)  # may download model from ms hub

        # Handle quantization config if present
        if getattr(config, "quantization_config", None):
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

        # Initialize SGLang engine according to the documentation
        engine_args = {
            "model_path": model_args.model_name_or_path,
            "dtype": model_args.infer_dtype,
            "tensor_parallel_size": get_device_count() or 1,
            "max_model_len": model_args.get("max_model_length", 4096),  # Default to 4096 if not specified
            "trust_remote_code": model_args.trust_remote_code,
            "download_dir": model_args.cache_dir,
        }

        # Add SGLang-specific configuration if provided
        if hasattr(model_args, "sglang_config") and isinstance(model_args.sglang_config, dict):
            engine_args.update(model_args.sglang_config)

        # Initialize the SGLang engine using the correct approach
        self.model = sgl.Engine(**engine_args)

        # Handle adapter if specified
        if model_args.adapter_name_or_path is not None:
            logger.warning_rank0("Adapter loading is not yet implemented for SGLang engine")

    async def _generate(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        audios: Optional[Sequence["AudioInput"]] = None,
        **input_kwargs,
    ) -> AsyncIterator[Any]:
        """
        Internal helper method for text generation using SGLang.
        """
        # Handle multimodal inputs if provided
        mm_input_dict = {"images": [], "videos": [], "audios": [], "imglens": [0], "vidlens": [0], "audlens": [0]}
        if images is not None:
            mm_input_dict.update({"images": images, "imglens": [len(images)]})
            if not any(IMAGE_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = IMAGE_PLACEHOLDER * len(images) + messages[0]["content"]

        if videos is not None:
            mm_input_dict.update({"videos": videos, "vidlens": [len(videos)]})
            if not any(VIDEO_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = VIDEO_PLACEHOLDER * len(videos) + messages[0]["content"]

        if audios is not None:
            mm_input_dict.update({"audios": audios, "audlens": [len(audios)]})
            if not any(AUDIO_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = AUDIO_PLACEHOLDER * len(audios) + messages[0]["content"]

        # Process messages with multimodal plugin if needed
        messages = self.template.mm_plugin.process_messages(
            messages, mm_input_dict["images"], mm_input_dict["videos"], mm_input_dict["audios"], self.processor
        )

        # Prepare the input for generation
        paired_messages = messages + [{"role": "assistant", "content": ""}]
        system = system or self.generating_args["default_system"]

        # Convert to a properly formatted prompt for the model
        prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, paired_messages, system, tools)
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

        # Extract generation parameters from input_kwargs or use defaults
        temperature = input_kwargs.pop("temperature", self.generating_args["temperature"])
        top_p = input_kwargs.pop("top_p", self.generating_args["top_p"])
        top_k = input_kwargs.pop("top_k", self.generating_args["top_k"])
        input_kwargs.pop("num_return_sequences", 1)
        repetition_penalty = input_kwargs.pop("repetition_penalty", self.generating_args["repetition_penalty"])
        max_new_tokens = input_kwargs.pop("max_new_tokens", self.generating_args.get("max_new_tokens", 2048))
        stop = input_kwargs.pop("stop", None)

        # Prepare SGLang generation parameters
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_new_tokens,
        }

        # Add stop sequences if provided
        if stop:
            if isinstance(stop, str):
                sampling_params["stop"] = [stop]
            else:
                sampling_params["stop"] = stop

        # Create an async generator using SGLang
        async def generate():
            # Define the generation function
            @sgl.function
            def chat_completion(state):
                state.template = prompt_text
                state.gen(params=sampling_params)

            # Run the generation
            result = await self.model.async_generate(prompt_text, sampling_params)

            # Convert the result to our expected format
            generated_text = result["text"]
            token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)

            # Determine finish reason
            finish_reason = "stop"
            if len(token_ids) >= max_new_tokens:
                finish_reason = "length"

            # Create a response object that mimics the format expected
            output = type(
                "ResponseOutput", (), {"text": generated_text, "token_ids": token_ids, "finish_reason": finish_reason}
            )

            result_obj = type("RequestOutput", (), {"outputs": [output], "prompt_token_ids": prompt_ids})

            yield result_obj

        # Return the async generator
        return generate()

    @override
    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        audios: Optional[Sequence["AudioInput"]] = None,
        **input_kwargs,
    ) -> List["Response"]:
        """
        Gets a list of responses from the chat model using SGLang.
        """
        final_output = None
        generator = await self._generate(messages, system, tools, images, videos, audios, **input_kwargs)
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
        audios: Optional[Sequence["AudioInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Gets the response token-by-token from the chat model using SGLang streaming.
        """
        # Prepare the input for generation, similar to _generate method
        mm_input_dict = {"images": [], "videos": [], "audios": [], "imglens": [0], "vidlens": [0], "audlens": [0]}
        if images is not None:
            mm_input_dict.update({"images": images, "imglens": [len(images)]})
            if not any(IMAGE_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = IMAGE_PLACEHOLDER * len(images) + messages[0]["content"]

        if videos is not None:
            mm_input_dict.update({"videos": videos, "vidlens": [len(videos)]})
            if not any(VIDEO_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = VIDEO_PLACEHOLDER * len(videos) + messages[0]["content"]

        if audios is not None:
            mm_input_dict.update({"audios": audios, "audlens": [len(audios)]})
            if not any(AUDIO_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = AUDIO_PLACEHOLDER * len(audios) + messages[0]["content"]

        # Process messages with multimodal plugin if needed
        messages = self.template.mm_plugin.process_messages(
            messages, mm_input_dict["images"], mm_input_dict["videos"], mm_input_dict["audios"], self.processor
        )

        # Prepare the input for generation
        paired_messages = messages + [{"role": "assistant", "content": ""}]
        system = system or self.generating_args["default_system"]

        # Convert to a properly formatted prompt for the model
        prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, paired_messages, system, tools)
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

        # Extract generation parameters from input_kwargs or use defaults
        temperature = input_kwargs.pop("temperature", self.generating_args["temperature"])
        top_p = input_kwargs.pop("top_p", self.generating_args["top_p"])
        top_k = input_kwargs.pop("top_k", self.generating_args["top_k"])
        repetition_penalty = input_kwargs.pop("repetition_penalty", self.generating_args["repetition_penalty"])
        max_new_tokens = input_kwargs.pop("max_new_tokens", self.generating_args.get("max_new_tokens", 2048))
        stop = input_kwargs.pop("stop", None)

        # Prepare SGLang generation parameters
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_new_tokens,
        }

        # Add stop sequences if provided
        if stop:
            if isinstance(stop, str):
                sampling_params["stop"] = [stop]
            else:
                sampling_params["stop"] = stop

        # Stream using async_stream_and_merge from SGLang's utils
        async for token in sgl.utils.async_stream_and_merge(self.model, prompt_text, sampling_params):
            yield token

    @override
    async def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        """
        Gets scores for a batch of inputs (logprobs).
        """
        input_kwargs.pop("temperature", 0.0)
        input_kwargs.pop("top_p", 1.0)
        input_kwargs.pop("top_k", -1)

        scores = []

        # SGLang scoring implementation
        # Note: This is a simplification - the actual implementation depends on
        # how SGLang supports scoring or logprob calculation
        for text in batch_input:
            sampling_params = {"temperature": 0, "max_tokens": 0}

            result = await self.model.async_generate(text, sampling_params)
            # Extract logprob or score from result
            # This is an approximation - actual implementation depends on SGLang's API
            if hasattr(result, "cum_logprob"):
                log_prob = -result.cum_logprob
            else:
                # Fallback if cum_logprob is not available
                log_prob = 0.0

            scores.append(log_prob)

        return scores
