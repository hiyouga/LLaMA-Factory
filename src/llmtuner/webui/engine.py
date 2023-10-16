import gradio as gr
from gradio.components import Component # cannot use TYPE_CHECKING here
from typing import Any, Dict, Generator, Optional

from llmtuner.webui.chatter import WebChatModel
from llmtuner.webui.common import get_model_path, list_dataset, load_config
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

    def _form_dict(self, resume_dict: Dict[str, Dict[str, Any]]):
        return {self.manager.get_elem(k): gr.update(**v) for k, v in resume_dict.items()}

    def resume(self) -> Generator[Dict[Component, Dict[str, Any]], None, None]:
        user_config = load_config()
        lang = user_config.get("lang", None) or "en"

        init_dict = {
            "top.lang": {"value": lang},
            "infer.chat_box": {"visible": self.chatter.loaded}
        }

        if not self.pure_chat:
            init_dict["train.dataset"] = {"choices": list_dataset()["choices"]}
            init_dict["eval.dataset"] = {"choices": list_dataset()["choices"]}

            if user_config.get("last_model", None):
                init_dict["top.model_name"] = {"value": user_config["last_model"]}
                init_dict["top.model_path"] = {"value": get_model_path(user_config["last_model"])}

        yield self._form_dict(init_dict)

        if not self.pure_chat:
            if self.runner.alive:
                yield {elem: gr.update(value=value) for elem, value in self.runner.data.items()}
                if self.runner.do_train:
                    yield self._form_dict({"train.resume_btn": {"value": True}})
                else:
                    yield self._form_dict({"eval.resume_btn": {"value": True}})
            else:
                yield self._form_dict({"train.output_dir": {"value": get_time()}})

    def change_lang(self, lang: str) -> Dict[Component, Dict[str, Any]]:
        return {
            component: gr.update(**LOCALES[name][lang])
            for elems in self.manager.all_elems.values() for name, component in elems.items() if name in LOCALES
        }
