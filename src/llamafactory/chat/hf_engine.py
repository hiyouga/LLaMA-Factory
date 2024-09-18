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

import asyncio
import concurrent.futures
import os
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import GenerationConfig, TextIteratorStreamer
from typing_extensions import override

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.logging import get_logger
from ..extras.misc import get_logits_processor
from ..model import load_model, load_tokenizer
from .base_engine import BaseEngine, Response


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from trl import PreTrainedModelWrapper

    from ..data import Template
    from ..data.mm_plugin import ImageInput, VideoInput
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


class HuggingfaceEngine(BaseEngine):
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
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
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            logger.warning("There is no current event loop, creating a new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT", "1")))

    @staticmethod
    def _process_args(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        template: "Template",
        generating_args: Dict[str, Any],
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["ImageInput"] = None,
        video: Optional["VideoInput"] = None,
        input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Tuple[Dict[str, Any], int]:
        mm_input_dict = {"images": [], "videos": [], "imglens": [0], "vidlens": [0]}
        if image is not None:
            mm_input_dict.update({"images": [image], "imglens": [1]})
            if IMAGE_PLACEHOLDER not in messages[0]["content"]:
                messages[0]["content"] = IMAGE_PLACEHOLDER + messages[0]["content"]

        if video is not None:
            mm_input_dict.update({"videos": [video], "vidlens": [1]})
            if VIDEO_PLACEHOLDER not in messages[0]["content"]:
                messages[0]["content"] = VIDEO_PLACEHOLDER + messages[0]["content"]

        messages = template.mm_plugin.process_messages(
            messages, mm_input_dict["images"], mm_input_dict["videos"], processor
        )
        paired_messages = messages + [{"role": "assistant", "content": ""}]
        system = system or generating_args["default_system"]
        prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, system, tools)
        prompt_ids, _ = template.mm_plugin.process_token_ids(
            prompt_ids, None, mm_input_dict["images"], mm_input_dict["videos"], tokenizer, processor
        )
        prompt_length = len(prompt_ids)
        inputs = torch.tensor([prompt_ids], device=model.device)
        attention_mask = torch.ones_like(inputs, dtype=torch.bool)

        do_sample: Optional[bool] = input_kwargs.pop("do_sample", None)
        temperature: Optional[float] = input_kwargs.pop("temperature", None)
        top_p: Optional[float] = input_kwargs.pop("top_p", None)
        top_k: Optional[float] = input_kwargs.pop("top_k", None)
        num_return_sequences: int = input_kwargs.pop("num_return_sequences", 1)
        repetition_penalty: Optional[float] = input_kwargs.pop("repetition_penalty", None)
        length_penalty: Optional[float] = input_kwargs.pop("length_penalty", None)
        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        max_new_tokens: Optional[int] = input_kwargs.pop("max_new_tokens", None)
        stop: Optional[Union[str, List[str]]] = input_kwargs.pop("stop", None)

        if stop is not None:
            logger.warning("Stop parameter is not supported by the huggingface engine yet.")

        generating_args = generating_args.copy()
        generating_args.update(
            dict(
                do_sample=do_sample if do_sample is not None else generating_args["do_sample"],
                temperature=temperature if temperature is not None else generating_args["temperature"],
                top_p=top_p if top_p is not None else generating_args["top_p"],
                top_k=top_k if top_k is not None else generating_args["top_k"],
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty
                if repetition_penalty is not None
                else generating_args["repetition_penalty"],
                length_penalty=length_penalty if length_penalty is not None else generating_args["length_penalty"],
                eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        )

        if isinstance(num_return_sequences, int) and num_return_sequences > 1:  # do_sample needs temperature > 0
            generating_args["do_sample"] = True
            generating_args["temperature"] = generating_args["temperature"] or 1.0

        if not generating_args["temperature"]:
            generating_args["do_sample"] = False

        if not generating_args["do_sample"]:
            generating_args.pop("temperature", None)
            generating_args.pop("top_p", None)

        if max_length:
            generating_args.pop("max_new_tokens", None)
            generating_args["max_length"] = max_length

        if max_new_tokens:
            generating_args.pop("max_length", None)
            generating_args["max_new_tokens"] = max_new_tokens

        gen_kwargs = dict(
            inputs=inputs,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(**generating_args),
            logits_processor=get_logits_processor(),
        )

        mm_inputs = template.mm_plugin.get_mm_inputs(**mm_input_dict, seqlens=[prompt_length], processor=processor)
        for key, value in mm_inputs.items():
            value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
            gen_kwargs[key] = value.to(model.device)

        return gen_kwargs, prompt_length

    @staticmethod
    @torch.inference_mode()
    def _chat(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        template: "Template",
        generating_args: Dict[str, Any],
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["ImageInput"] = None,
        video: Optional["VideoInput"] = None,
        input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> List["Response"]:
        gen_kwargs, prompt_length = HuggingfaceEngine._process_args(
            model, tokenizer, processor, template, generating_args, messages, system, tools, image, video, input_kwargs
        )
        generate_output = model.generate(**gen_kwargs)
        response_ids = generate_output[:, prompt_length:]
        response = tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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
        generating_args: Dict[str, Any],
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["ImageInput"] = None,
        video: Optional["VideoInput"] = None,
        input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Callable[[], str]:
        gen_kwargs, _ = HuggingfaceEngine._process_args(
            model, tokenizer, processor, template, generating_args, messages, system, tools, image, video, input_kwargs
        )
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
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
        batch_input: List[str],
        input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> List[float]:
        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        device = getattr(model.pretrained_model, "device", "cuda")
        inputs: Dict[str, "torch.Tensor"] = tokenizer(
            batch_input,
            padding=True,
            truncation=True,
            max_length=max_length or getattr(model.config, "max_position_embeddings", 1024),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        values: "torch.Tensor" = model(**inputs, return_dict=True, use_cache=False)[-1]
        scores = values.gather(dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return scores

    @override
    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["ImageInput"] = None,
        video: Optional["VideoInput"] = None,
        **input_kwargs,
    ) -> List["Response"]:
        if not self.can_generate:
            raise ValueError("The current model does not support `chat`.")

        loop = asyncio.get_running_loop()
        input_args = (
            self.model,
            self.tokenizer,
            self.processor,
            self.template,
            self.generating_args,
            messages,
            system,
            tools,
            image,
            video,
            input_kwargs,
        )
        async with self.semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, self._chat, *input_args)

    @override
    async def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["ImageInput"] = None,
        video: Optional["VideoInput"] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        if not self.can_generate:
            raise ValueError("The current model does not support `stream_chat`.")

        loop = asyncio.get_running_loop()
        input_args = (
            self.model,
            self.tokenizer,
            self.processor,
            self.template,
            self.generating_args,
            messages,
            system,
            tools,
            image,
            video,
            input_kwargs,
        )
        async with self.semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                stream = self._stream_chat(*input_args)
                while True:
                    try:
                        yield await loop.run_in_executor(pool, stream)
                    except StopAsyncIteration:
                        break

    @override
    async def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        if self.can_generate:
            raise ValueError("Cannot get scores using an auto-regressive model.")

        loop = asyncio.get_running_loop()
        input_args = (self.model, self.tokenizer, batch_input, input_kwargs)
        async with self.semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, self._get_scores, *input_args)
