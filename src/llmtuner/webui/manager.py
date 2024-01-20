from typing import TYPE_CHECKING, Dict, List, Set


if TYPE_CHECKING:
    from gradio.components import Component


class Manager:
    def __init__(self) -> None:
        self.all_elems: Dict[str, Dict[str, "Component"]] = {}

    def get_elem_by_name(self, name: str) -> "Component":
        r"""
        Example: top.lang, train.dataset
        """
        tab_name, elem_name = name.split(".")
        return self.all_elems[tab_name][elem_name]

    def get_base_elems(self) -> Set["Component"]:
        return {
            self.all_elems["top"]["lang"],
            self.all_elems["top"]["model_name"],
            self.all_elems["top"]["model_path"],
            self.all_elems["top"]["adapter_path"],
            self.all_elems["top"]["finetuning_type"],
            self.all_elems["top"]["quantization_bit"],
            self.all_elems["top"]["template"],
            self.all_elems["top"]["rope_scaling"],
            self.all_elems["top"]["booster"],
        }

    def list_elems(self) -> List["Component"]:
        return [elem for elems in self.all_elems.values() for elem in elems.values()]
