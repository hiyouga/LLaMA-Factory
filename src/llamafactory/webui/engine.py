from typing import TYPE_CHECKING, Any, Dict

from .chatter import WebChatModel
from .common import get_model_path, list_dataset, load_config
from .locales import LOCALES
from .manager import Manager
from .runner import Runner
from .utils import get_time


if TYPE_CHECKING:
    from gradio.components import Component


class Engine:
    def __init__(self, demo_mode: bool = False, pure_chat: bool = False) -> None:
        self.demo_mode = demo_mode
        self.pure_chat = pure_chat
        self.manager = Manager()
        self.runner = Runner(self.manager, demo_mode)
        self.chatter = WebChatModel(self.manager, demo_mode, lazy_init=(not pure_chat))

    def _update_component(self, input_dict: Dict[str, Dict[str, Any]]) -> Dict["Component", "Component"]:
        r"""
        Gets the dict to update the components.
        """
        output_dict: Dict["Component", "Component"] = {}
        for elem_id, elem_attr in input_dict.items():
            elem = self.manager.get_elem_by_id(elem_id)
            output_dict[elem] = elem.__class__(**elem_attr)

        return output_dict

    def resume(self):
        user_config = load_config() if not self.demo_mode else {}
        lang = user_config.get("lang", None) or "en"

        init_dict = {"top.lang": {"value": lang}, "infer.chat_box": {"visible": self.chatter.loaded}}

        if not self.pure_chat:
            init_dict["train.dataset"] = {"choices": list_dataset().choices}
            init_dict["eval.dataset"] = {"choices": list_dataset().choices}
            init_dict["train.output_dir"] = {"value": "train_{}".format(get_time())}
            init_dict["train.config_path"] = {"value": "{}.yaml".format(get_time())}
            init_dict["eval.output_dir"] = {"value": "eval_{}".format(get_time())}
            init_dict["infer.image_box"] = {"visible": False}

            if user_config.get("last_model", None):
                init_dict["top.model_name"] = {"value": user_config["last_model"]}
                init_dict["top.model_path"] = {"value": get_model_path(user_config["last_model"])}

        yield self._update_component(init_dict)

        if self.runner.running and not self.demo_mode and not self.pure_chat:
            yield {elem: elem.__class__(value=value) for elem, value in self.runner.running_data.items()}
            if self.runner.do_train:
                yield self._update_component({"train.resume_btn": {"value": True}})
            else:
                yield self._update_component({"eval.resume_btn": {"value": True}})

    def change_lang(self, lang: str):
        return {
            elem: elem.__class__(**LOCALES[elem_name][lang])
            for elem_name, elem in self.manager.get_elem_iter()
            if elem_name in LOCALES
        }
