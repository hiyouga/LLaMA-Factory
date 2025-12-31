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

"""The definition of kernel registry.

Init Phase:
1. Define kernel registry.
2. Register kernels.

"""

from typing import Optional

from ....accelerator.helper import get_current_accelerator
from .base import BaseKernel


__all__ = ["Registry", "register_kernel"]


class Registry:
    r"""Registry for managing kernel implementations.

    Storage structure: ``{ "kernel_id": Class }``
    """

    _kernels: dict[str, type[BaseKernel]] = {}

    @classmethod
    def register(cls, kernel_cls: type[BaseKernel]):
        r"""Decorator to register a kernel class.

        The class must inherit from :class:`BaseKernel` and specify ``_kernel_id`` and ``_device`` attributes.

        Args:
            kernel_cls (type[BaseKernel]): The kernel class to register.

        Returns:
            type[BaseKernel]: The registered kernel class.

        Raises:
            TypeError: If the class does not inherit from :class:`BaseKernel`.
            ValueError: If the kernel ID is missing or already registered.
        """
        if not issubclass(kernel_cls, BaseKernel):
            raise TypeError(f"Class {kernel_cls} must inherit from BaseKernel")
        kernel_id = kernel_cls.get_kernel_id()
        device = kernel_cls.get_device()

        # The device type of the current accelerator does not match the device type required by the kernel, skip registration
        if device != get_current_accelerator().type:
            return

        if not kernel_id:
            raise ValueError(f"Kernel ID (_kernel_id) is needed for {kernel_cls} to register")

        if kernel_id in cls._kernels:
            raise ValueError(f"{kernel_id} already registered! The registered kernel is {cls._kernels[kernel_id]}")

        cls._kernels[kernel_id] = kernel_cls
        return kernel_cls

    @classmethod
    def get(cls, kernel_id: str) -> Optional[type[BaseKernel]]:
        r"""Retrieves a registered kernel implementation by its ID.

        Args:
            kernel_id (str): The ID of the kernel to retrieve.

        Returns:
            Optional[type[BaseKernel]]: The kernel class if found, else ``None``.
        """
        return cls._kernels.get(kernel_id)

    @classmethod
    def get_registered_kernels(cls) -> dict[str, type[BaseKernel]]:
        r"""Returns a dictionary of all registered kernels.

        Returns:
            dict[str, type[BaseKernel]]: Dictionary mapping kernel IDs to kernel classes.
        """
        return cls._kernels


# export decorator alias
register_kernel = Registry.register
