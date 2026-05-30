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

"""Kernel implementation base classes.

``BaseKernel``
    Template base for every concrete kernel implementation.  Subclasses
    declare ``_device`` and implement ``_apply()``.  ``check_deps()`` and
    the ``apply()`` template method are inherited for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ....utils.plugin import ensure_methods_implemented


if TYPE_CHECKING:
    from ....accelerator.helper import DeviceType
    from ....utils.types import HFModel


class BaseKernel(ABC):
    """Base class for all kernel implementations.

    Subclasses must declare ``_device`` and implement ``_apply()``.
    Override ``check_deps()`` to add richer validation (package version,
    hardware flags, …) on top of the default device-type check.
    """

    _device: DeviceType

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        ensure_methods_implemented(cls)

    @classmethod
    def check_deps(cls) -> None:
        """Verify that the current accelerator matches ``_device``."""
        from ....accelerator.helper import get_current_accelerator

        current = get_current_accelerator().type
        if cls._device != current:
            raise RuntimeError(f"Kernel {cls.__name__!r} requires {cls._device}, current accelerator is {current}.")

    @classmethod
    def apply(cls, **kwargs) -> HFModel:
        """Template method: check deps then delegate to ``_apply()``."""
        cls.check_deps()
        return cls._apply(**kwargs)

    @classmethod
    @abstractmethod
    def _apply(cls, **kwargs) -> HFModel:
        """Kernel implementation — override in each concrete subclass."""
