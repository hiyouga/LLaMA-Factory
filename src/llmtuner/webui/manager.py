import gradio as gr
from gradio.components import Component
from typing import Any, Dict, List

from llmtuner.webui.common import get_model_path, list_dataset, load_config
from llmtuner.webui.locales import LOCALES
from llmtuner.webui.utils import get_time


class Manager:

    def __init__(self, elem_list: List[Dict[str, Component]]):
        self.elem_list = elem_list

    def gen_refresh(self, lang: str) -> Dict[str, Any]:
        refresh_dict = {
            "dataset": {"choices": list_dataset()["choices"]},
            "output_dir": {"value": get_time()}
        }

        user_config = load_config()
        if not lang:
            if user_config.get("lang", None):
                lang = user_config["lang"]
            else:
                lang = "en"

        refresh_dict["lang"] = {"value": lang}

        if user_config.get("last_model", None):
            refresh_dict["model_name"] = {"value": user_config["last_model"]}
            refresh_dict["model_path"] = {"value": get_model_path(user_config["last_model"])}

        return refresh_dict

    def gen_label(self, lang: str) -> Dict[Component, Dict[str, Any]]: # cannot use TYPE_CHECKING
        update_dict = {}
        refresh_dict = self.gen_refresh(lang)

        for elems in self.elem_list:
            for name, component in elems.items():
                update_dict[component] = gr.update(
                    **LOCALES[name][refresh_dict["lang"]["value"]], **refresh_dict.get(name, {})
                )

        return update_dict
