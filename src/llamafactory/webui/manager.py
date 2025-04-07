# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Generator
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from gradio.components import Component


class Manager:
    r"""A class to manage all the gradio components in Web UI."""

<<<<<<< HEAD
    def add_elems(self, tab_name: str, elem_dict: Dict[str, "Component"]) -> None:
        r"""Adds elements to manager."""
        # print(elem_dict.items())
=======
    def __init__(self) -> None:
        self._id_to_elem: dict[str, Component] = {}
        self._elem_to_id: dict[Component, str] = {}

    def add_elems(self, tab_name: str, elem_dict: dict[str, "Component"]) -> None:
        r"""Add elements to manager."""
>>>>>>> 7e0cdb1a76c1ac6b69de86d4ba40e9395f883cdb
        for elem_name, elem in elem_dict.items():
            elem_id = f"{tab_name}.{elem_name}"
            self._id_to_elem[elem_id] = elem
            self._elem_to_id[elem] = elem_id

<<<<<<< HEAD
    def get_elem_list(self) -> List["Component"]:
        r"""Returns the list of all elements."""
        return list(self._id_to_elem.values())

    def get_elem_iter(self) -> Generator[Tuple[str, "Component"], None, None]:
        r"""Returns an iterator over all elements with their names."""
=======
    def get_elem_list(self) -> list["Component"]:
        r"""Return the list of all elements."""
        return list(self._id_to_elem.values())

    def get_elem_iter(self) -> Generator[tuple[str, "Component"], None, None]:
        r"""Return an iterator over all elements with their names."""
>>>>>>> 7e0cdb1a76c1ac6b69de86d4ba40e9395f883cdb
        for elem_id, elem in self._id_to_elem.items():
            yield elem_id.split(".")[-1], elem

    def get_elem_by_id(self, elem_id: str) -> "Component":
<<<<<<< HEAD
        r"""Gets element by id.
=======
        r"""Get element by id.
>>>>>>> 7e0cdb1a76c1ac6b69de86d4ba40e9395f883cdb

        Example: top.lang, train.dataset
        """
        # print(self._id_to_elem)
        return self._id_to_elem[elem_id]

    def get_id_by_elem(self, elem: "Component") -> str:
<<<<<<< HEAD
        r"""Gets id by element."""
        return self._elem_to_id[elem]

    def get_base_elems(self) -> Set["Component"]:
        r"""Gets the base elements that are commonly used."""
=======
        r"""Get id by element."""
        return self._elem_to_id[elem]

    def get_base_elems(self) -> set["Component"]:
        r"""Get the base elements that are commonly used."""
>>>>>>> 7e0cdb1a76c1ac6b69de86d4ba40e9395f883cdb
        return {
            self._id_to_elem["top.lang"],
            self._id_to_elem["top.model_name"],
            self._id_to_elem["top.model_path"],
            self._id_to_elem["top.finetuning_type"],
            self._id_to_elem["top.checkpoint_path"],
            self._id_to_elem["top.quantization_bit"],
            self._id_to_elem["top.quantization_method"],
            self._id_to_elem["top.template"],
            self._id_to_elem["top.rope_scaling"],
            self._id_to_elem["top.booster"],
        }
