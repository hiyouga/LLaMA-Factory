import asyncio
import gc
import os
import re
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

import psutil

# Import SGLang correctly
import sglang as sgl
import torch
from typing_extensions import override

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import (
    AUDIO_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    VIDEO_PLACEHOLDER,
)
from ..extras.logging import get_logger
from ..hparams import (
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    ModelArguments,
)
from ..model import load_config, load_tokenizer
from ..model.model_utils.quantization import QuantizationMethod
from .base_engine import BaseEngine, Response


if TYPE_CHECKING:
    from ..data import AudioInput, ImageInput, VideoInput


logger = get_logger(__name__)


class SGLangEngine(BaseEngine):
    """
    Inference engine for SGLang models.

    This class wraps the SGLang engine to provide a consistent interface for text generation
    that matches LLaMA Factory's requirements. It uses the offline SGLang engine (not the HTTP server)
    and focuses on the core parameters needed for generation.
    """

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
        try:
            # Clean memory before initialization
            self._clean_memory()
            self._initialize_engine(model_args, data_args, finetuning_args, generating_args)
        except Exception as e:
            logger.error(f"Failed to initialize SGLang engine: {str(e)}")
            logger.error("Make sure that SGLang is installed correctly and configured properly")
            raise

    def _clean_memory(self):
        """Free memory before initializing the engine"""
        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache before SGLang initialization")
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache: {str(e)}")

        # Log memory stats
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            logger.info(f"Memory usage before SGLang initialization: {mem_info.rss / (1024 * 1024):.2f} MB")

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logger.info(
                        f"CUDA memory usage on device {i}: "
                        f"{torch.cuda.memory_allocated(i) / (1024 * 1024):.2f} MB allocated, "
                        f"{torch.cuda.memory_reserved(i) / (1024 * 1024):.2f} MB reserved"
                    )
        except Exception:
            logger.warning("Could not log memory statistics")

    def _initialize_engine(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        """Internal method to initialize the SGLang engine with proper error handling"""
        # Log SGLang version for debugging
        try:
            sgl_version = getattr(sgl, "__version__", "unknown")
            logger.info(f"Using SGLang version: {sgl_version}")
        except Exception:
            logger.warning("Unable to determine SGLang version")

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

        # Memory conservation: Try to free up memory before model loading
        self._clean_memory()

        # Initialize SGLang engine with minimal parameters
        engine_args = {"model_path": model_args.model_name_or_path, "dtype": model_args.infer_dtype}

        # Only add essential parameters
        if model_args.cache_dir:
            engine_args["download_dir"] = model_args.cache_dir

        logger.info(f"Initializing SGLang engine with args: {engine_args}")

        # Initialize the SGLang engine with robust error handling
        try:
            # Attempt to initialize the engine
            self.model = sgl.Engine(**engine_args)
            logger.info("SGLang engine initialized successfully")
        except TypeError as e:
            # Special handling for TypeError that indicates unsupported parameters
            error_message = str(e)
            if "unexpected keyword argument" in error_message:
                param_match = re.search(r"unexpected keyword argument '([^']+)'", error_message)
                if param_match:
                    bad_param = param_match.group(1)
                    logger.warning(f"SGLang doesn't support parameter: '{bad_param}', removing it")
                    engine_args.pop(bad_param, None)

                    # Try again with the parameter removed
                    try:
                        self.model = sgl.Engine(**engine_args)
                        logger.info("SGLang engine initialized successfully after removing unsupported parameter")
                        return
                    except Exception as inner_e:
                        logger.error(f"Still failed after removing parameter: {str(inner_e)}")
                        raise

            logger.error(f"SGLang engine initialization failed: {error_message}")
            logger.error("Make sure to only use parameters supported by your SGLang version")
            raise
        except Exception as e:
            logger.error(f"SGLang engine initialization failed: {str(e)}")
            logger.error("Try setting CUDA_VISIBLE_DEVICES=0 to use only one GPU")
            logger.error("Also ensure flashinfer is properly installed if you want to use that backend")
            raise

        # Handle adapter if specified
        if model_args.adapter_name_or_path is not None:
            logger.warning("Adapter loading is not yet implemented for SGLang engine")

    async def _generate(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        # Need to explore more about multimodal inputs
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
        input_kwargs.pop("num_return_sequences", 1)  # Still pop it but don't assign to variable
        repetition_penalty = input_kwargs.pop("repetition_penalty", self.generating_args["repetition_penalty"])
        max_new_tokens = input_kwargs.pop("max_new_tokens", self.generating_args.get("max_new_tokens", 2048))
        stop = input_kwargs.pop("stop", None)

        # Prepare minimal SGLang generation parameters
        sampling_params = {"max_new_tokens": max_new_tokens}

        # Only add non-default parameters
        if temperature > 0:
            sampling_params["temperature"] = temperature
        if top_p < 1.0:
            sampling_params["top_p"] = top_p
        if top_k > 0:
            sampling_params["top_k"] = top_k
        if repetition_penalty != 1.0:
            sampling_params["repetition_penalty"] = repetition_penalty

        # Add stop sequences if provided
        if stop:
            if isinstance(stop, str):
                sampling_params["stop"] = [stop]
            else:
                sampling_params["stop"] = stop

        # Create an async generator to yield results
        async def generate():
            try:
                # Generate using SGLang's API
                logger.info(f"Generating with prompt: {prompt_text[:100]}...")

                # Try different API methods with proper error handling
                result = None
                try:
                    # First try the async method if available
                    if hasattr(self.model, "generate_async"):
                        result = await self.model.generate_async([prompt_text], sampling_params)
                    elif hasattr(self.model, "async_generate"):  # Alternative name in some versions
                        result = await self.model.async_generate([prompt_text], sampling_params)
                    else:
                        # Fallback to non-async API
                        result = await asyncio.to_thread(self.model.generate, [prompt_text], sampling_params)
                except Exception as e:
                    logger.warning(f"Error with initial generation attempt: {str(e)}, trying minimal params")
                    # Try with minimal parameters
                    result = await asyncio.to_thread(
                        self.model.generate, [prompt_text], {"max_new_tokens": max_new_tokens}
                    )

                if not result:
                    raise RuntimeError("Generation failed with no result")

                # Extract the result
                if isinstance(result, list) and len(result) > 0:
                    if "text" in result[0]:
                        generated_text = result[0]["text"]
                    else:
                        # Try to find the output in various response formats
                        generated_text = str(result[0])
                else:
                    # Fallback for other result types
                    generated_text = str(result)

                token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)

                # Determine finish reason
                finish_reason = "stop"
                if len(token_ids) >= max_new_tokens:
                    finish_reason = "length"

                # Create a response object
                output = type(
                    "ResponseOutput",
                    (),
                    {"text": generated_text, "token_ids": token_ids, "finish_reason": finish_reason},
                )

                result_obj = type("RequestOutput", (), {"outputs": [output], "prompt_token_ids": prompt_ids})

                yield result_obj
            except Exception as e:
                logger.error(f"Error in SGLang generation: {str(e)}")
                raise

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
        Streams responses from the chat model token by token.
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
        prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, paired_messages, system, tools)
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

        # Extract generation parameters from input_kwargs or use defaults
        temperature = input_kwargs.pop("temperature", self.generating_args["temperature"])
        top_p = input_kwargs.pop("top_p", self.generating_args["top_p"])
        top_k = input_kwargs.pop("top_k", self.generating_args["top_k"])
        repetition_penalty = input_kwargs.pop("repetition_penalty", self.generating_args["repetition_penalty"])
        max_new_tokens = input_kwargs.pop("max_new_tokens", self.generating_args.get("max_new_tokens", 2048))
        stop = input_kwargs.pop("stop", None)

        # Prepare minimal SGLang generation parameters
        sampling_params = {"max_new_tokens": max_new_tokens}

        # Only add non-default parameters
        if temperature > 0:
            sampling_params["temperature"] = temperature
        if top_p < 1.0:
            sampling_params["top_p"] = top_p
        if top_k > 0:
            sampling_params["top_k"] = top_k
        if repetition_penalty != 1.0:
            sampling_params["repetition_penalty"] = repetition_penalty

        # Add stop sequences if provided
        if stop:
            if isinstance(stop, str):
                sampling_params["stop"] = [stop]
            else:
                sampling_params["stop"] = stop

        # Set stream to true for streaming
        sampling_params["stream"] = True

        try:
            # Use SGLang's streaming API with proper fallbacks
            logger.info(f"Streaming with prompt: {prompt_text[:100]}...")

            # Try different possible streaming methods
            streaming_method = None
            if hasattr(self.model, "generate_stream_async"):
                streaming_method = self.model.generate_stream_async
            elif hasattr(self.model, "generate_stream"):
                # Wrap in asyncio.to_thread to make non-async methods work in async context
                streaming_method = lambda text, params: asyncio.to_thread(self.model.generate_stream, text, params)
            else:
                # Fallback to non-streaming generation
                logger.warning("Streaming API not available, falling back to regular generation")
                result = await asyncio.to_thread(self.model.generate, [prompt_text], sampling_params)
                yield result[0]["text"]
                return

            # Generate stream
            stream_iterator = await streaming_method([prompt_text], sampling_params)

            # Process streaming output based on different possible formats
            async for output in stream_iterator:
                if isinstance(output, dict):
                    if "token" in output and isinstance(output["token"], dict) and "text" in output["token"]:
                        # Handle {"token": {"text": ["token1", "token2"]}}
                        for token in output["token"]["text"]:
                            yield token
                    elif "text" in output:
                        # Handle {"text": "token"}
                        yield output["text"]
                    elif "content" in output:
                        # Handle {"content": "token"} alternative format
                        yield output["content"]
                    elif "delta" in output and "content" in output["delta"]:
                        # Handle ChatCompletion-like format
                        yield output["delta"]["content"]
                elif isinstance(output, str):
                    # Direct string output
                    yield output
                else:
                    # Try to convert unknown formats to string
                    try:
                        yield str(output)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error in SGLang streaming: {str(e)}")
            # Try fallback to non-streaming generation
            try:
                sampling_params["stream"] = False
                result = await asyncio.to_thread(self.model.generate, [prompt_text], sampling_params)
                yield result[0]["text"]
            except Exception as e2:
                logger.error(f"Fallback generation also failed: {str(e2)}")
                raise e

    @override
    async def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        """
        Gets scores for a batch of inputs (logprobs).

        Note: SGLang doesn't have a direct scoring API, so we use generation
        with max_tokens=0 to get the logprobs without generating any text.
        """
        input_kwargs.pop("temperature", 0.0)
        input_kwargs.pop("top_p", 1.0)
        input_kwargs.pop("top_k", -1)

        scores = []

        try:
            # Minimal parameters for scoring only
            sampling_params = {"max_new_tokens": 0}

            # Process each input separately with robust error handling
            for text in batch_input:
                try:
                    # Try different API methods
                    if hasattr(self.model, "generate_async"):
                        result = await self.model.generate_async([text], sampling_params)
                    elif hasattr(self.model, "async_generate"):
                        result = await self.model.async_generate([text], sampling_params)
                    else:
                        # Fallback to non-async API
                        result = await asyncio.to_thread(self.model.generate, [text], sampling_params)

                    # Default score
                    log_prob = 0.0

                    # Extract score from various possible result formats
                    if isinstance(result, list) and len(result) > 0:
                        item = result[0]
                        if isinstance(item, dict):
                            # Try different field names for logprobs
                            for field in ["logprob", "cum_logprob", "log_prob", "score"]:
                                if field in item:
                                    try:
                                        val = item[field]
                                        log_prob = -float(val) if val is not None else 0.0
                                        break
                                    except (ValueError, TypeError):
                                        # Handle non-numeric values
                                        continue

                    scores.append(log_prob)
                except Exception as e:
                    logger.warning(f"Error scoring input: {str(e)}")
                    scores.append(0.0)  # Default score for failed items

            return scores
        except Exception as e:
            logger.error(f"Error in SGLang scoring: {str(e)}")
            # Return default scores in case of error
            return [0.0] * len(batch_input)
