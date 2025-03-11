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

    def __init__(self) -> None:
        self._id_to_elem: dict[str, Component] = {}
        self._elem_to_id: dict[Component, str] = {}

    def add_elems(self, tab_name: str, elem_dict: dict[str, "Component"]) -> None:
        r"""Add elements to manager."""
        for elem_name, elem in elem_dict.items():
            elem_id = f"{tab_name}.{elem_name}"
            self._id_to_elem[elem_id] = elem
            self._elem_to_id[elem] = elem_id

    def get_elem_list(self) -> list["Component"]:
        r"""Return the list of all elements."""
        return list(self._id_to_elem.values())

    def get_elem_iter(self) -> Generator[tuple[str, "Component"], None, None]:
        r"""Return an iterator over all elements with their names."""
        for elem_id, elem in self._id_to_elem.items():
            yield elem_id.split(".")[-1], elem

    def get_elem_by_id(self, elem_id: str) -> "Component":
        r"""Get element by id.

        Example: top.lang, train.dataset
        """
        return self._id_to_elem[elem_id]

    def get_id_by_elem(self, elem: "Component") -> str:
        r"""Get id by element."""
        return self._elem_to_id[elem]

    def get_base_elems(self) -> set["Component"]:
        r"""Get the base elements that are commonly used."""
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
