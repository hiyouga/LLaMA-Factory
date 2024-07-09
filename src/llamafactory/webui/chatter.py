import json
import os
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Sequence, Tuple

from numpy.typing import NDArray

from ..chat import ChatModel
from ..data import Role
from ..extras.constants import PEFT_METHODS
from ..extras.misc import torch_gc
from ..extras.packages import is_gradio_available
from .common import get_save_dir
from .locales import ALERTS


if TYPE_CHECKING:
    from ..chat import BaseEngine
    from .manager import Manager


if is_gradio_available():
    import gradio as gr


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
            finetuning_type=finetuning_type,
            quantization_bit=int(get("top.quantization_bit")) if get("top.quantization_bit") in ["8", "4"] else None,
            template=get("top.template"),
            flash_attn="fa2" if get("top.booster") == "flashattn2" else "auto",
            use_unsloth=(get("top.booster") == "unsloth"),
            visual_inputs=get("top.visual_inputs"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") in ["linear", "dynamic"] else None,
            infer_backend=get("infer.infer_backend"),
        )

        if checkpoint_path:
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in checkpoint_path]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, checkpoint_path)

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

    def append(
        self,
        chatbot: List[List[Optional[str]]],
        messages: Sequence[Dict[str, str]],
        role: str,
        query: str,
    ) -> Tuple[List[List[Optional[str]]], List[Dict[str, str]], str]:
        return chatbot + [[query, None]], messages + [{"role": role, "content": query}], ""

    def stream(
        self,
        chatbot: List[List[Optional[str]]],
        messages: Sequence[Dict[str, str]],
        system: str,
        tools: str,
        image: Optional[NDArray],
        max_new_tokens: int,
        top_p: float,
        temperature: float,
    ) -> Generator[Tuple[List[List[Optional[str]]], List[Dict[str, str]]], None, None]:
        chatbot[-1][1] = ""
        response = ""
        for new_text in self.stream_chat(
            messages, system, tools, image, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature
        ):
            response += new_text
            if tools:
                result = self.engine.template.format_tools.extract(response)
            else:
                result = response

            if isinstance(result, tuple):
                name, arguments = result
                arguments = json.loads(arguments)
                tool_call = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
                output_messages = messages + [{"role": Role.FUNCTION.value, "content": tool_call}]
                bot_text = "```json\n" + tool_call + "\n```"
            else:
                output_messages = messages + [{"role": Role.ASSISTANT.value, "content": result}]
                bot_text = result

            chatbot[-1][1] = bot_text
            yield chatbot, output_messages
