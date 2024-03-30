from typing import TYPE_CHECKING, Dict, Generator, List, Set, Tuple


if TYPE_CHECKING:
    from gradio.components import Component


class Manager:
    def __init__(self) -> None:
        self._elem_dicts: Dict[str, Dict[str, "Component"]] = {}

    def add_elem_dict(self, tab_name: str, elem_dict: Dict[str, "Component"]) -> None:
        r"""
        Adds a elem dict.
        """
        self._elem_dicts[tab_name] = elem_dict

    def get_elem_list(self) -> List["Component"]:
        r"""
        Returns the list of all elements.
        """
        return [elem for elem_dict in self._elem_dicts.values() for elem in elem_dict.values()]

    def get_elem_iter(self) -> Generator[Tuple[str, "Component"], None, None]:
        r"""
        Returns an iterator over all elements with their names.
        """
        for elem_dict in self._elem_dicts.values():
            for elem_name, elem in elem_dict.items():
                yield elem_name, elem

    def get_elem_by_id(self, elem_id: str) -> "Component":
        r"""
        Gets element by id.

        Example: top.lang, train.dataset
        """
        tab_name, elem_name = elem_id.split(".")
        return self._elem_dicts[tab_name][elem_name]

    def get_base_elems(self) -> Set["Component"]:
        r"""
        Gets the base elements that are commonly used.
        """
        return {
            self._elem_dicts["top"]["lang"],
            self._elem_dicts["top"]["model_name"],
            self._elem_dicts["top"]["model_path"],
            self._elem_dicts["top"]["finetuning_type"],
            self._elem_dicts["top"]["adapter_path"],
            self._elem_dicts["top"]["quantization_bit"],
            self._elem_dicts["top"]["template"],
            self._elem_dicts["top"]["rope_scaling"],
            self._elem_dicts["top"]["booster"],
        }
