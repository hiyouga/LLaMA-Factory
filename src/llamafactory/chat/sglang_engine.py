import asyncio
import gc
import os
import re
import subprocess
import time
import requests
import atexit
import json
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
        """Initialize a SGLang server and connect to it via HTTP"""
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

        # Find an available port for the server
        self.port = find_available_port()
        self.host = "127.0.0.1"
        self.base_url = f"http://{self.host}:{self.port}"

        # Prepare SGLang server launch command
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_args.model_name_or_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        # Add dtype if specified
        if model_args.infer_dtype and model_args.infer_dtype != "auto":
            cmd.extend(["--dtype", model_args.infer_dtype])

        # Add cache dir if specified
        if model_args.cache_dir:
            cmd.extend(["--download-dir", model_args.cache_dir])

        # Launch the server process
        logger.info(f"Starting SGLang server with command: {' '.join(cmd)}")
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Register cleanup handler to terminate server when Python exits
            atexit.register(self._cleanup_server)

            # Wait for server to be ready
            server_ready = wait_for_server(self.base_url, timeout=60)
            if not server_ready:
                raise RuntimeError("Timed out waiting for SGLang server to start")

            # Get model info to verify server is working correctly
            try:
                response = requests.get(f"{self.base_url}/get_model_info")
                if response.status_code == 200:
                    model_info = response.json()
                    logger.info(f"SGLang server model info: {model_info}")
                else:
                    logger.warning(f"Failed to get model info: {response.status_code}")
            except Exception as e:
                logger.warning(f"Error getting model info: {str(e)}")

            logger.info(f"SGLang server initialized successfully at {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to start SGLang server: {str(e)}")
            self._cleanup_server()  # Make sure to clean up any started process
            raise RuntimeError(f"SGLang server initialization failed: {str(e)}")

        # Handle adapter if specified
        if model_args.adapter_name_or_path is not None:
            logger.warning("Adapter loading is not yet implemented for SGLang engine")

    def _cleanup_server(self):
        """Clean up the server process when the engine is destroyed"""
        if hasattr(self, "server_process") and self.server_process:
            try:
                logger.info("Terminating SGLang server process")
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                logger.info("SGLang server process terminated")
            except Exception as e:
                logger.warning(f"Error terminating SGLang server: {str(e)}")
                try:
                    self.server_process.kill()
                except Exception:
                    pass

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
        Internal helper method for text generation using SGLang server.
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

        # Prepare sampling parameters for SGLang
        sampling_params = {
            "temperature": temperature if temperature > 0 else 0.01,
            "top_p": top_p if top_p < 1.0 else 0.95,
            "max_new_tokens": max_new_tokens,
        }

        # Add additional parameters
        if top_k > 0:
            sampling_params["top_k"] = top_k
        if repetition_penalty != 1.0:
            sampling_params["repetition_penalty"] = repetition_penalty
        if stop:
            sampling_params["stop"] = [stop] if isinstance(stop, str) else stop

        # Create an async generator to yield results
        async def generate():
            try:
                # Generate using SGLang HTTP API
                logger.info(f"Generating with prompt: {prompt_text[:100]}...")

                # Prepare request data
                request_data = {"text": prompt_text, "sampling_params": sampling_params}

                # Make HTTP request to SGLang server
                response = await asyncio.to_thread(
                    requests.post, f"{self.base_url}/generate", json=request_data
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"SGLang server error: {response.status_code}, {response.text}"
                    )

                response_data = response.json()

                # Process the response
                if "text" in response_data:
                    generated_text = response_data["text"]
                    meta_info = response_data.get("meta_info", {})
                    finish_reason = "stop"

                    # Extract finish reason if available
                    if "finish_reason" in meta_info:
                        fr = meta_info["finish_reason"]
                        if isinstance(fr, dict) and "type" in fr:
                            finish_reason = fr["type"]
                        elif isinstance(fr, str):
                            finish_reason = fr
                    elif len(generated_text) >= max_new_tokens:
                        finish_reason = "length"

                    # Encode text to get token IDs
                    token_ids = self.tokenizer.encode(
                        generated_text, add_special_tokens=False
                    )

                    # Create response objects
                    output = type(
                        "ResponseOutput",
                        (),
                        {
                            "text": generated_text,
                            "token_ids": token_ids,
                            "finish_reason": finish_reason,
                        },
                    )

                    result_obj = type(
                        "RequestOutput",
                        (),
                        {"outputs": [output], "prompt_token_ids": prompt_ids},
                    )

                    yield result_obj
                else:
                    raise RuntimeError(f"Unexpected response format: {response_data}")

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
        Streams responses from the chat model token by token via SGLang server.
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

        # Prepare sampling parameters for SGLang
        sampling_params = {
            "temperature": temperature if temperature > 0 else 0.01,
            "top_p": top_p if top_p < 1.0 else 0.95,
            "max_new_tokens": max_new_tokens,
        }

        # Add additional parameters
        if top_k > 0:
            sampling_params["top_k"] = top_k
        if repetition_penalty != 1.0:
            sampling_params["repetition_penalty"] = repetition_penalty
        if stop:
            sampling_params["stop"] = [stop] if isinstance(stop, str) else stop

        request_data = {
            "text": prompt_text,
            "sampling_params": sampling_params,
            "stream": True,
        }

        logger.info(f"Streaming with prompt: {prompt_text[:100]}...")

        try:
            # We need to run this in a thread since it's blocking I/O
            def stream_request():
                response = requests.post(
                    f"{self.base_url}/generate",
                    json=request_data,
                    stream=True,  # Enable streaming on the request
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"SGLang server error: {response.status_code}, {response.text}"
                    )

                # Stream processing similar to the example
                text_so_far = ""
                for chunk in response.iter_lines(decode_unicode=False):
                    if not chunk:
                        continue

                    chunk = chunk.decode("utf-8")

                    # Handle streaming format from SGLang server
                    if chunk.startswith("data:"):
                        if chunk == "data: [DONE]":
                            break

                        try:
                            data = json.loads(chunk[5:].strip("\n"))
                            if "text" in data:
                                current_text = data["text"].strip()
                                new_text = current_text[len(text_so_far) :]
                                text_so_far = current_text
                                yield new_text
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse chunk as JSON: {chunk}, error: {str(e)}"
                            )
                            # Try to extract any text content if possible
                            if "text" in chunk:
                                try:
                                    start = chunk.find('"text":') + 8
                                    end = chunk.find('",', start)
                                    if end == -1:
                                        end = chunk.find('"}', start)
                                    if start > 0 and end > start:
                                        text = chunk[start:end].strip('"')
                                        yield text
                                except Exception:
                                    pass

                # Return any remaining text if available
                if text_so_far:
                    pass  # We already yielded all text incrementally

            # Run the streaming request in a thread and process responses
            stream_gen = await asyncio.to_thread(stream_request)
            for text in stream_gen:
                yield text

        except Exception as e:
            logger.error(f"Error in SGLang streaming: {str(e)}")
            # Yield an error message as a last resort
            yield f"[Error in text generation: {str(e)}]"

    @override
    async def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        """
        Gets scores for a batch of inputs (logprobs) using the SGLang server.
        """
        input_kwargs.pop("temperature", 0.0)
        input_kwargs.pop("top_p", 1.0)
        input_kwargs.pop("top_k", -1)

        scores = []

        try:
            # Process each input separately with robust error handling
            for text in batch_input:
                try:
                    # Prepare minimal parameters for scoring only
                    request_data = {
                        "text": text,
                        "max_new_tokens": 0,  # No generation, just scoring
                    }

                    # Send request to server
                    response = await asyncio.to_thread(
                        requests.post, f"{self.base_url}/generate", json=request_data
                    )

                    if response.status_code != 200:
                        logger.warning(
                            f"Error scoring input: status code {response.status_code}"
                        )
                        scores.append(0.0)
                        continue

                    result = response.json()

                    # Default score
                    log_prob = 0.0

                    # Extract score from meta_info if available
                    if "meta_info" in result:
                        meta_info = result["meta_info"]
                        for field in ["logprob", "cum_logprob", "log_prob", "score"]:
                            if field in meta_info:
                                try:
                                    val = meta_info[field]
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

    def __del__(self):
        """Ensure server is cleaned up when object is deleted"""
        self._cleanup_server()
        # Also unregister from atexit to avoid duplicate cleanup
        try:
            atexit.unregister(self._cleanup_server)
        except Exception:
            pass


def find_available_port():
    """Find an available port on the local machine."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_for_server(url, timeout=120, interval=1.0):
    """Wait for the SGLang server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                logger.info(f"SGLang server is ready at {url}")
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)

    logger.error(f"Timed out waiting for SGLang server at {url}")
    return False
