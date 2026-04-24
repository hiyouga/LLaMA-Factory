# Copyright 2025 the KVCache.AI team, Approaching AI, and the LlamaFactory team.
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
import platform
from collections.abc import AsyncGenerator
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

import torch
from typing_extensions import override

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import EngineName
from ..model import load_model, load_tokenizer
from .base_engine import BaseEngine, Response


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from trl import PreTrainedModelWrapper

    from ..data.mm_plugin import AudioInput, ImageInput, VideoInput
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.server.config.config import Config
from ktransformers.util.utils import (
    get_compute_capability,
    prefill_and_generate_capture,
)
from ktransformers.util.vendors import GPUVendor, device_manager


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

        tok_mod = load_tokenizer(model_args)
        self.tokenizer = tok_mod["tokenizer"]
        self.tokenizer.padding_side = "left" if self.can_generate else "right"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)

        self.model = load_model(
            self.tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=(not self.can_generate)
        )

        self.generating_args = generating_args.to_dict()
        self.max_new_tokens = model_args.kt_maxlen
        self.use_cuda_graph = model_args.kt_use_cuda_graph
        self.mode = model_args.kt_mode
        self.force_think = model_args.kt_force_think
        self.chunk_size = model_args.chunk_size

        try:
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT", "1")))

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
        inputs = tokenizer(
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

    async def _generate(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        paired = messages + [{"role": "assistant", "content": ""}]
        prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, paired, system, tools)
        prompt_len = len(prompt_ids)

        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        max_new_tokens: Optional[int] = input_kwargs.pop("max_new_tokens", None)

        if "max_new_tokens" in self.generating_args:
            max_tokens = int(self.generating_args["max_new_tokens"])
        elif "max_length" in self.generating_args:
            gl = int(self.generating_args["max_length"])
            max_tokens = gl - prompt_len if gl > prompt_len else 1
        else:
            max_tokens = self.max_new_tokens or 256

        if max_length is not None:
            max_tokens = max(max_length - prompt_len, 1)
        if max_new_tokens is not None:
            max_tokens = int(max_new_tokens)
        max_tokens = max(1, int(max_tokens))

        if self.mode == "long_context":
            max_len_cfg = Config().long_context_config["max_seq_len"]
            need = prompt_len + max_tokens
            assert max_len_cfg > need, f"please set max_seq_len > {need} in ~/.ktransformers/config.yaml"

        device = next(self.model.parameters()).device
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if self.force_think:
            think = torch.tensor(
                [self.tokenizer.encode("<think>\n", add_special_tokens=False)], dtype=torch.long, device=device
            )
            input_tensor = torch.cat([input_tensor, think], dim=1)

        use_flashinfer = (
            platform.system() != "Windows"
            and getattr(self.model.config, "architectures", [""])[0]
            in {"DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"}
            and flashinfer_enabled
            and get_compute_capability() >= 8
            and device_manager.gpu_vendor == GPUVendor.NVIDIA
        )

        def make_gen():
            if use_flashinfer:
                return prefill_and_generate_capture(
                    self.model,
                    self.tokenizer,
                    input_tensor,
                    max_tokens,
                    self.use_cuda_graph,
                    mode=self.mode,
                    force_think=self.force_think,
                    chunk_size=self.chunk_size,
                    use_flashinfer_mla=True,
                    num_heads=self.model.config.num_attention_heads,
                    head_dim_ckv=getattr(self.model.config, "kv_lora_rank", 0),
                    head_dim_kpe=getattr(self.model.config, "qk_rope_head_dim", 0),
                    q_head_dim=getattr(self.model.config, "qk_rope_head_dim", 0)
                    + getattr(self.model.config, "qk_nope_head_dim", 0),
                    echo_stream=False,
                )
            else:
                return prefill_and_generate_capture(
                    self.model,
                    self.tokenizer,
                    input_tensor,
                    max_tokens,
                    self.use_cuda_graph,
                    mode=self.mode,
                    force_think=self.force_think,
                    chunk_size=self.chunk_size,
                    echo_stream=False,
                )

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[str]] = asyncio.Queue()

        def producer():
            try:
                gen = make_gen()
                if hasattr(gen, "__aiter__"):

                    async def drain_async():
                        async for t in gen:
                            loop.call_soon_threadsafe(q.put_nowait, t if isinstance(t, str) else str(t))

                    asyncio.run(drain_async())
                elif hasattr(gen, "__iter__"):
                    for t in gen:
                        loop.call_soon_threadsafe(q.put_nowait, t if isinstance(t, str) else str(t))
                else:
                    loop.call_soon_threadsafe(q.put_nowait, gen if isinstance(gen, str) else str(gen))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        Thread(target=producer, daemon=True).start()

        while True:
            item = await q.get()
            if item is None:
                break
            yield item

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
        async with self.semaphore:
            produced = ""
            final_text = ""
            async for t in self._generate(messages, system, tools, **input_kwargs):
                delta = t
                produced = produced + delta
                if delta:
                    final_text += delta

            prompt_ids, _ = self.template.encode_oneturn(
                self.tokenizer, messages + [{"role": "assistant", "content": ""}], system, tools
            )
            return [
                Response(
                    response_text=final_text,
                    response_length=len(self.tokenizer.encode(final_text, add_special_tokens=False)),
                    prompt_length=len(prompt_ids),
                    finish_reason="stop",
                )
            ]

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
        async with self.semaphore:
            produced = ""
            async for t in self._generate(messages, system, tools, **input_kwargs):
                delta = t[len(produced) :] if t.startswith(produced) else t
                produced = t
                if delta:
                    yield delta

    @override
    async def get_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        if self.can_generate:
            raise ValueError("Cannot get scores using an auto-regressive model.")
        args = (self.model, self.tokenizer, batch_input, input_kwargs)
        async with self.semaphore:
            return await asyncio.to_thread(self._get_scores, *args)
