# Copyright 2025 the LlamaFactory team.
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

import asyncio
import os
from collections.abc import AsyncGenerator
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import platform

import torch
from transformers import GenerationConfig, TextIteratorStreamer
from typing_extensions import override

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import AUDIO_PLACEHOLDER, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER, EngineName
from ..model import load_model, load_tokenizer
from .base_engine import BaseEngine, Response

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from trl import PreTrainedModelWrapper

    from ..data import Template
    from ..data.mm_plugin import AudioInput, ImageInput, VideoInput
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

from ktransformers.util.utils import load_weights, prefill_and_generate, prefill_and_generate_capture, get_compute_capability, xpu_fp16_model
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.util.vendors import device_manager, get_device, to_device, GPUVendor
from ktransformers.server.config.config import Config

logger = logging.get_logger(__name__)


class KTransformersEngine(BaseEngine):
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.name = EngineName.KT
        self.can_generate = finetuning_args.stage == "sft"
        tokenizer_module = load_tokenizer(model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        self.tokenizer.padding_side = "left" if self.can_generate else "right"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        self.model = load_model(
            self.tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=(not self.can_generate)
        )  # must after fixing tokenizer to resize vocab
        self.generating_args = generating_args.to_dict()
        self.max_new_tokens = model_args.kt_maxlen
        self.use_cuda_graph = model_args.kt_use_cuda_graph
        self.mode = model_args.kt_mode
        self.force_think = model_args.kt_force_think
        self.chunk_size = model_args.chunk_size
        
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            logger.warning_rank0_once("There is no current event loop, creating a new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT", "1")))

    @staticmethod
    @torch.inference_mode()
    def _chat(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        template: "Template",
        generating_args: dict[str, Any],
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        input_kwargs: Optional[dict[str, Any]] = {},
    ) -> list["Response"]:
        generate_output = model.generate(**gen_kwargs)
        if isinstance(generate_output, tuple):
            generate_output = generate_output[1][0]  # post-process the minicpm_o output

        response_ids = generate_output[:, prompt_length:]
        response = tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=getattr(gen_kwargs["generation_config"], "skip_special_tokens", True),
            clean_up_tokenization_spaces=True,
        )
        results = []
        for i in range(len(response)):
            eos_index = (response_ids[i] == tokenizer.eos_token_id).nonzero()
            response_length = (eos_index[0].item() + 1) if len(eos_index) else len(response_ids[i])
            results.append(
                Response(
                    response_text=response[i],
                    response_length=response_length,
                    prompt_length=prompt_length,
                    finish_reason="stop" if len(eos_index) else "length",
                )
            )

        return results

    @staticmethod
    @torch.inference_mode()
    def _stream_chat(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        template: "Template",
        generating_args: dict[str, Any],
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        input_kwargs: Optional[dict[str, Any]] = {},
    ) -> Callable[[], str]:
        gen_kwargs, _ = HuggingfaceEngine._process_args(
            model,
            tokenizer,
            processor,
            template,
            generating_args,
            messages,
            system,
            tools,
            images,
            videos,
            audios,
            input_kwargs,
        )
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=getattr(gen_kwargs["generation_config"], "skip_special_tokens", True),
        )
        gen_kwargs["streamer"] = streamer
        thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        def stream():
            try:
                return streamer.__next__()
            except StopIteration:
                raise StopAsyncIteration()

        return stream

    @staticmethod
    @torch.inference_mode()
    def _get_scores(
        model: "PreTrainedModelWrapper",
        tokenizer: "PreTrainedTokenizer",
        batch_input: list[str],
        input_kwargs: Optional[dict[str, Any]] = {},
    ) -> list[float]:
        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        device = getattr(model.pretrained_model, "device", "cuda")
        inputs: dict[str, torch.Tensor] = tokenizer(
            batch_input,
            padding=True,
            truncation=True,
            max_length=max_length or getattr(model.config, "max_position_embeddings", 1024),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        values: torch.Tensor = model(**inputs, return_dict=True, use_cache=False)[-1]
        scores = values.gather(dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return scores

    @override
    async def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        if not self.can_generate:
            raise ValueError("The current model does not support `chat`.")

        input_args = (
            self.model,
            self.tokenizer,
            self.processor,
            self.template,
            self.generating_args,
            messages,
            system,
            tools,
            images,
            videos,
            audios,
            input_kwargs,
        )
        async with self.semaphore:
            return await asyncio.to_thread(self._chat, *input_args)

    @override
    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        if not self.can_generate:
            raise ValueError("The current model does not support `stream_chat`.")
        prompt_messages = messages if not system else ([{"role": "system", "content": system}] + messages)
        input_tensor = self.tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, return_tensors="pt"
        )
        config = self.model.config
        device = next(self.model.parameters()).device
        if self.force_think:
            token_thinks = torch.tensor([self.tokenizer.encode("<think>\n", add_special_tokens=False)], device=input_tensor.device)
            input_tensor = torch.cat([input_tensor, token_thinks], dim=1)
        if self.mode == "long_context":
            assert Config().long_context_config["max_seq_len"] > input_tensor.shape[1] + self.max_new_tokens, "please change max_seq_len in  ~/.ktransformers/config.yaml"
        use_flashinfer_path = (
            platform.system() != "Windows"
            and getattr(config, "architectures", [""])[0] in {"DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"}
            and flashinfer_enabled
            and get_compute_capability() >= 8
            and device_manager.gpu_vendor == GPUVendor.NVIDIA
        )
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        SENTINEL = object()
        def producer():
            try:
                if use_flashinfer_path:
                    gen = prefill_and_generate_capture(
                        self.model,
                        self.tokenizer,
                        input_tensor.to(device),
                        self.max_new_tokens,
                        self.use_cuda_graph,
                        mode=self.mode,
                        force_think=self.force_think,
                        chunk_size=self.chunk_size,
                        use_flashinfer_mla=True,
                        num_heads=config.num_attention_heads,
                        head_dim_ckv=config.kv_lora_rank,
                        head_dim_kpe=config.qk_rope_head_dim,
                        q_head_dim=config.qk_rope_head_dim + config.qk_nope_head_dim, 
                        echo_stream=False
                    )
                else:
                    gen = prefill_and_generate_capture(
                        self.model,
                        self.tokenizer,
                        input_tensor.to(device),
                        self.max_new_tokens,
                        self.use_cuda_graph,
                        mode=self.mode,
                        force_think=self.force_think,
                        chunk_size=self.chunk_size, 
                        echo_stream=False
                    )
                if hasattr(gen, "__aiter__"):
                    async def drain_async():
                        async for t in gen:
                            loop.call_soon_threadsafe(q.put_nowait, t)
                    asyncio.run(drain_async())
                elif hasattr(gen, "__iter__"):
                    for t in gen:
                        loop.call_soon_threadsafe(q.put_nowait, t)
                else:
                    loop.call_soon_threadsafe(q.put_nowait, str(gen))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)
        Thread(target=producer, daemon=True).start()
        while True:
            item = await q.get()
            if item is None:
                break
            yield item


    @override
    async def get_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        if self.can_generate:
            raise ValueError("Cannot get scores using an auto-regressive model.")

        input_args = (self.model, self.tokenizer, batch_input, input_kwargs)
        async with self.semaphore:
            return await asyncio.to_thread(self._get_scores, *input_args)
