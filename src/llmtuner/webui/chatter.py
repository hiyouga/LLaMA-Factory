import gradio as gr
from gradio.components import Component # cannot use TYPE_CHECKING here
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple

from llmtuner.chat import ChatModel
from llmtuner.extras.misc import torch_gc
from llmtuner.hparams import GeneratingArguments
from llmtuner.webui.common import get_save_dir
from llmtuner.webui.locales import ALERTS

if TYPE_CHECKING:
    from llmtuner.webui.manager import Manager


class WebChatModel(ChatModel):

    def __init__(
        self,
        manager: "Manager",
        demo_mode: Optional[bool] = False,
        lazy_init: Optional[bool] = True
    ) -> None:
        self.manager = manager
        self.demo_mode = demo_mode
        self.model = None
        self.tokenizer = None
        self.generating_args = GeneratingArguments()

        if not lazy_init: # read arguments from command line
            super().__init__()

        if demo_mode: # load demo_config.json if exists
            import json
            try:
                with open("demo_config.json", "r", encoding="utf-8") as f:
                    args = json.load(f)
                super().__init__(args)
            except:
                print("Cannot find `demo_config.json` at current directory.")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load_model(self, data: Dict[Component, Any]) -> Generator[str, None, None]:
        get = lambda name: data[self.manager.get_elem_by_name(name)]
        lang = get("top.lang")
        error = ""
        if self.loaded:
            error = ALERTS["err_exists"][lang]
        elif not get("top.model_name"):
            error = ALERTS["err_no_model"][lang]
        elif not get("top.model_path"):
            error = ALERTS["err_no_path"][lang]
        elif self.demo_mode:
            error = ALERTS["err_demo"][lang]

        if error:
            gr.Warning(error)
            yield error
            return

        if get("top.checkpoints"):
            checkpoint_dir = ",".join([
                get_save_dir(get("top.model_name"), get("top.finetuning_type"), ckpt) for ckpt in get("top.checkpoints")
            ])
        else:
            checkpoint_dir = None

        yield ALERTS["info_loading"][lang]
        args = dict(
            model_name_or_path=get("top.model_path"),
            checkpoint_dir=checkpoint_dir,
            finetuning_type=get("top.finetuning_type"),
            quantization_bit=int(get("top.quantization_bit")) if get("top.quantization_bit") in ["8", "4"] else None,
            template=get("top.template"),
            system_prompt=get("top.system_prompt"),
            flash_attn=get("top.flash_attn"),
            shift_attn=get("top.shift_attn"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") in ["linear", "dynamic"] else None
        )
        super().__init__(args)

        yield ALERTS["info_loaded"][lang]

    def unload_model(self, data: Dict[Component, Any]) -> Generator[str, None, None]:
        lang = data[self.manager.get_elem_by_name("top.lang")]

        if self.demo_mode:
            yield ALERTS["err_demo"][lang]
            return

        yield ALERTS["info_unloading"][lang]
        self.model = None
        self.tokenizer = None
        torch_gc()
        yield ALERTS["info_unloaded"][lang]

    def predict(
        self,
        chatbot: List[Tuple[str, str]],
        query: str,
        history: List[Tuple[str, str]],
        system: str,
        max_new_tokens: int,
        top_p: float,
        temperature: float
    ) -> Generator[Tuple[List[Tuple[str, str]], List[Tuple[str, str]]], None, None]:
        chatbot.append([query, ""])
        response = ""
        for new_text in self.stream_chat(
            query, history, system, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature
        ):
            response += new_text
            new_history = history + [(query, response)]
            chatbot[-1] = [query, self.postprocess(response)]
            yield chatbot, new_history

    def postprocess(self, response: str) -> str:
        blocks = response.split("```")
        for i, block in enumerate(blocks):
            if i % 2 == 0:
                blocks[i] = block.replace("<", "&lt;").replace(">", "&gt;")
        return "```".join(blocks)
