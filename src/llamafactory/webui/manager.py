from typing import TYPE_CHECKING, Dict, Generator, List, Set, Tuple


if TYPE_CHECKING:
    from gradio.components import Component


class Manager:
    def __init__(self) -> None:
        self._id_to_elem: Dict[str, "Component"] = {}
        self._elem_to_id: Dict["Component", str] = {}

    def add_elems(self, tab_name: str, elem_dict: Dict[str, "Component"]) -> None:
        r"""
        Adds elements to manager.
        """
        for elem_name, elem in elem_dict.items():
            elem_id = "{}.{}".format(tab_name, elem_name)
            self._id_to_elem[elem_id] = elem
            self._elem_to_id[elem] = elem_id

    def get_elem_list(self) -> List["Component"]:
        r"""
        Returns the list of all elements.
        """
        return list(self._id_to_elem.values())

    def get_elem_iter(self) -> Generator[Tuple[str, "Component"], None, None]:
        r"""
        Returns an iterator over all elements with their names.
        """
        for elem_id, elem in self._id_to_elem.items():
            yield elem_id.split(".")[-1], elem

    def get_elem_by_id(self, elem_id: str) -> "Component":
        r"""
        Gets element by id.

        Example: top.lang, train.dataset
        """
        return self._id_to_elem[elem_id]

    def get_id_by_elem(self, elem: "Component") -> str:
        r"""
        Gets id by element.
        """
        return self._elem_to_id[elem]

    def get_base_elems(self) -> Set["Component"]:
        r"""
        Gets the base elements that are commonly used.
        """
        return {
            self._id_to_elem["top.lang"],
            self._id_to_elem["top.model_name"],
            self._id_to_elem["top.model_path"],
            self._id_to_elem["top.finetuning_type"],
            self._id_to_elem["top.adapter_path"],
            self._id_to_elem["top.quantization_bit"],
            self._id_to_elem["top.template"],
            self._id_to_elem["top.rope_scaling"],
            self._id_to_elem["top.booster"],
            self._id_to_elem["top.visual_inputs"],
        }
