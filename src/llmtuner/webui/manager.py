import gradio as gr
from typing import Any, Dict, List
from gradio.components import Component

from llmtuner.webui.common import get_model_path, list_dataset, load_config
from llmtuner.webui.locales import LOCALES
from llmtuner.webui.utils import get_time


class Manager:

    def __init__(self, elem_list: List[Dict[str, Component]]):
        self.elem_list = elem_list

    def gen_refresh(self) -> Dict[str, Any]:
        refresh_dict = {
            "dataset": {"choices": list_dataset()["choices"]},
            "output_dir": {"value": get_time()}
        }
        user_config = load_config()
        if user_config["last_model"]:
            refresh_dict["model_name"] = {"value": user_config["last_model"]}
            refresh_dict["model_path"] = {"value": get_model_path(user_config["last_model"])}

        return refresh_dict

    def gen_label(self, lang: str) -> Dict[Component, dict]:
        update_dict = {}
        refresh_dict = self.gen_refresh()

        for elems in self.elem_list:
            for name, component in elems.items():
                update_dict[component] = gr.update(**LOCALES[name][lang], **refresh_dict.get(name, {}))

        return update_dict
