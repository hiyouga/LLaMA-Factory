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

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple

from transformers.utils import is_torch_npu_available

from ..chat import ChatModel
from ..data import Role
from ..extras.constants import PEFT_METHODS
from ..extras.misc import torch_gc
from ..extras.packages import is_gradio_available
from .common import get_save_dir, load_config
from .locales import ALERTS


if TYPE_CHECKING:
    from ..chat import BaseEngine
    from .manager import Manager


if is_gradio_available():
    import gradio as gr


def _format_response(text: str, lang: str, thought_words: Tuple[str, str] = ("<think>", "</think>")) -> str:
    r"""
    Post-processes the response text.

    Based on: https://huggingface.co/spaces/Lyte/DeepSeek-R1-Distill-Qwen-1.5B-Demo-GGUF/blob/main/app.py
    """
    if thought_words[0] not in text:
        return text

    text = text.replace(thought_words[0], "")
    result = text.split(thought_words[1], maxsplit=1)
    if len(result) == 1:
        summary = ALERTS["info_thinking"][lang]
        thought, answer = text, ""
    else:
        summary = ALERTS["info_thought"][lang]
        thought, answer = result

    return (
        f"<details open><summary class='thinking-summary'><span>{summary}</span></summary>\n\n"
        f"<div class='thinking-container'>\n{thought}\n</div>\n</details>{answer}"
    )


class WebChatModel(ChatModel):
    def __init__(self, manager: "Manager", demo_mode: bool = False, lazy_init: bool = True) -> None:
        self.manager = manager
        self.demo_mode = demo_mode
        self.engine: Optional["BaseEngine"] = None

        if not lazy_init:  # read arguments from command line
            super().__init__()

        if demo_mode and os.environ.get("DEMO_MODEL") and os.environ.get("DEMO_TEMPLATE"):  # load demo model
            model_name_or_path = os.environ.get("DEMO_MODEL")
            template = os.environ.get("DEMO_TEMPLATE")
            infer_backend = os.environ.get("DEMO_BACKEND", "huggingface")
            super().__init__(
                dict(model_name_or_path=model_name_or_path, template=template, infer_backend=infer_backend)
            )

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def load_model(self, data) -> Generator[str, None, None]:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, model_path = get("top.lang"), get("top.model_name"), get("top.model_path")
        finetuning_type, checkpoint_path = get("top.finetuning_type"), get("top.checkpoint_path")
        user_config = load_config()

        error = ""
        if self.loaded:
            error = ALERTS["err_exists"][lang]
        elif not model_name:
            error = ALERTS["err_no_model"][lang]
        elif not model_path:
            error = ALERTS["err_no_path"][lang]
        elif self.demo_mode:
            error = ALERTS["err_demo"][lang]

        if error:
            gr.Warning(error)
            yield error
            return

        yield ALERTS["info_loading"][lang]
        args = dict(
            model_name_or_path=model_path,
            cache_dir=user_config.get("cache_dir", None),
            finetuning_type=finetuning_type,
            template=get("top.template"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") != "none" else None,
            flash_attn="fa2" if get("top.booster") == "flashattn2" else "auto",
            use_unsloth=(get("top.booster") == "unsloth"),
            enable_liger_kernel=(get("top.booster") == "liger_kernel"),
            infer_backend=get("infer.infer_backend"),
            infer_dtype=get("infer.infer_dtype"),
            trust_remote_code=True,
        )

        # checkpoints
        if checkpoint_path:
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in checkpoint_path]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, checkpoint_path)

        # quantization
        if get("top.quantization_bit") != "none":
            args["quantization_bit"] = int(get("top.quantization_bit"))
            args["quantization_method"] = get("top.quantization_method")
            args["double_quantization"] = not is_torch_npu_available()

        super().__init__(args)
        yield ALERTS["info_loaded"][lang]

    def unload_model(self, data) -> Generator[str, None, None]:
        lang = data[self.manager.get_elem_by_id("top.lang")]

        if self.demo_mode:
            gr.Warning(ALERTS["err_demo"][lang])
            yield ALERTS["err_demo"][lang]
            return

        yield ALERTS["info_unloading"][lang]
        self.engine = None
        torch_gc()
        yield ALERTS["info_unloaded"][lang]

    @staticmethod
    def append(
        chatbot: List[Dict[str, str]],
        messages: List[Dict[str, str]],
        role: str,
        query: str,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
        r"""
        Adds the user input to chatbot.

        Inputs: infer.chatbot, infer.messages, infer.role, infer.query
        Output: infer.chatbot, infer.messages
        """
        return chatbot + [{"role": "user", "content": query}], messages + [{"role": role, "content": query}], ""

    def stream(
        self,
        chatbot: List[Dict[str, str]],
        messages: List[Dict[str, str]],
        lang: str,
        system: str,
        tools: str,
        image: Optional[Any],
        video: Optional[Any],
        audio: Optional[Any],
        max_new_tokens: int,
        top_p: float,
        temperature: float,
    ) -> Generator[Tuple[List[Dict[str, str]], List[Dict[str, str]]], None, None]:
        r"""
        Generates output text in stream.

        Inputs: infer.chatbot, infer.messages, infer.system, infer.tools, infer.image, infer.video, ...
        Output: infer.chatbot, infer.messages
        """
        chatbot.append({"role": "assistant", "content": ""})
        response = ""
        for new_text in self.stream_chat(
            messages,
            system,
            tools,
            images=[image] if image else None,
            videos=[video] if video else None,
            audios=[audio] if audio else None,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        ):
            response += new_text
            if tools:
                result = self.engine.template.extract_tool(response)
            else:
                result = response

            if isinstance(result, list):
                tool_calls = [{"name": tool.name, "arguments": json.loads(tool.arguments)} for tool in result]
                tool_calls = json.dumps(tool_calls, ensure_ascii=False)
                output_messages = messages + [{"role": Role.FUNCTION.value, "content": tool_calls}]
                bot_text = "```json\n" + tool_calls + "\n```"
            else:
                output_messages = messages + [{"role": Role.ASSISTANT.value, "content": result}]
                bot_text = _format_response(result, lang, self.engine.template.thought_words)

            chatbot[-1] = {"role": "assistant", "content": bot_text}
            yield chatbot, output_messages
