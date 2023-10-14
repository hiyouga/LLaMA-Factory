import gradio as gr
from gradio.components import Component # cannot use TYPE_CHECKING here
from typing import Any, Dict, Generator, List, Optional, Tuple

from llmtuner.webui.chatter import WebChatModel
from llmtuner.webui.common import get_model_path, list_dataset, CONFIG_CLASS
from llmtuner.webui.locales import LOCALES
from llmtuner.webui.manager import Manager
from llmtuner.webui.runner import Runner
from llmtuner.webui.utils import get_time


class Engine:

    def __init__(self, pure_chat: Optional[bool] = False) -> None:
        self.pure_chat = pure_chat
        self.manager: "Manager" = Manager()
        self.runner: "Runner" = Runner(self.manager)
        self.chatter: "WebChatModel" = WebChatModel(manager=self.manager, lazy_init=(not pure_chat))

    def resume(self, config: CONFIG_CLASS) -> Generator[Dict[Component, Dict[str, Any]], None, None]:
        lang = config.get("lang", None) or "en"

        resume_dict = {
            "top.lang": {"value": lang},
            "infer.chat_box": {"visible": self.chatter.loaded}
        }

        if not self.pure_chat:
            resume_dict["train.dataset"] = {"choices": list_dataset()["choices"]}
            resume_dict["eval.dataset"] = {"choices": list_dataset()["choices"]}

            if config.get("last_model", None):
                resume_dict["top.model_name"] = {"value": config["last_model"]}
                resume_dict["top.model_path"] = {"value": get_model_path(config, config["last_model"])}

        yield {self.manager.get_elem(k): gr.update(**v) for k, v in resume_dict.items()}

        if self.runner.alive:
            pass # TODO: restore training
        else:
            resume_dict = {"train.output_dir": {"value": get_time()}} # TODO: xxx

    def change_lang(self, lang: str) -> Dict[Component, Dict[str, Any]]:
        return {
            component: gr.update(**LOCALES[name][lang])
            for elems in self.manager.all_elems.values() for name, component in elems.items() if name in LOCALES
        }
