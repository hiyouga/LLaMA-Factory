import gradio as gr
from gradio.components import Component
from typing import Any, Dict, List

from llmtuner.webui.common import get_model_path, list_dataset, load_config
from llmtuner.webui.locales import LOCALES
from llmtuner.webui.utils import get_time
from llmtuner.extras.constants import TRAINING_STAGES, DEFAULT_MODULE


class Manager:

    def __init__(self, elem_list: List[Dict[str, Component]]):
        self.elem_list = elem_list

    def gen_refresh(self, lang: str, model_name: str, finetuning_type: str, training_stage: str) -> Dict[str, Any]:
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

        if user_config.get("last_model", None) and not model_name:
            refresh_dict["model_name"] = {"value": user_config["last_model"]}
            refresh_dict["model_path"] = {"value": get_model_path(user_config["last_model"])}

        refresh_dict["lang"] = {"value": lang}
        refresh_dict["lora_tab"] = {"visible": finetuning_type == "lora"}

        training_stage = TRAINING_STAGES[training_stage]
        refresh_dict["rlhf_tab"] = {"visible": training_stage not in ["pt", "sft", "rm"]}
        refresh_dict["dpo_beta"] = {"visible": training_stage in ["dpo"]}
        refresh_dict["reward_model"] = {"visible": training_stage in ["ppo"]}
        refresh_dict["refresh_btn"] = {"visible": training_stage in ["ppo"]}
        if model_name.split("-")[0] in DEFAULT_MODULE:
            refresh_dict["lora_target"] = {"value": DEFAULT_MODULE[model_name.split("-")[0]]}

        return refresh_dict

    def gen_label(self, lang: str, model_name: str, finetuning_type: str = "lora", training_stage: str = list(TRAINING_STAGES.keys())[0]
    ) -> Dict[Component, Dict[str, Any]]: # cannot use TYPE_CHECKING
        update_dict = {}
        refresh_dict = self.gen_refresh(lang, model_name, finetuning_type, training_stage)
        for elems in self.elem_list:
            for name, component in elems.items():
                update_dict[component] = gr.update(
                    **LOCALES[name][refresh_dict["lang"]["value"]], **refresh_dict.get(name, {})
                )

        return update_dict
