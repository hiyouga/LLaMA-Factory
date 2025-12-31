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

"""The definition of base kernel class.

Init Phase:
1. Define base kernel class.
2. Define abstract methods.

"""

from abc import ABC, abstractmethod
from typing import Any

from ....accelerator.helper import DeviceType, get_current_accelerator
from ....utils.types import HFModel


class BaseKernel(ABC):
    r"""Base class for all kernel implementations.

    Subclasses must implement the abstract methods and define the required class attributes.
    """

    _kernel_id: Any = ""  # kernel ID, any hashable value to identify a kernel implementation
    _device: DeviceType = DeviceType.CPU  # "cuda", "npu", "cpu", etc.

    @classmethod
    def get_kernel_id(cls) -> str:
        r"""Returns the unique identifier for the kernel."""
        return cls._kernel_id

    @classmethod
    def get_device(cls) -> str:
        r"""Returns the device type associated with the kernel (e.g., "cuda", "npu", "cpu")."""
        return cls._device

    @classmethod
    def check_deps(cls) -> bool:
        r"""Checks if the required dependencies for the kernel are available.

        Returns:
            bool: ``True`` if dependencies are met, ``False`` otherwise.

        .. note::
            In explicit mode, if a user specifies an implementation but this check fails,
            it should raise an error instead of silently switching.
            Kernels can override this method to implement custom dependency checks.
        """
        if cls._device != get_current_accelerator().type:
            return False
        return True

    @classmethod
    @abstractmethod
    def apply(cls, **kwargs) -> HFModel:
        r"""Applies the kernel optimization to the model.

        Args:
            **kwargs: Arbitrary keyword arguments, usually containing the model instance and the kernel configuration.

        Returns:
            HFModel: The model with the kernel applied.

        Raises:
            RuntimeError: If the kernel dependencies are not met.
            NotImplementedError: If the method is not implemented by the subclass.

        Example:
            >>> from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_kernel
            >>> model = HFModel(config=config)
            >>> model = apply_kernel(model=model, kernel_id="npu_fused_moe")
        """
        if not cls.check_deps():
            raise RuntimeError(f"{cls.__name__} is not available but {cls.__name__} kernel was called.")
        raise NotImplementedError
