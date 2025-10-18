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

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from ....extras.types import HFModel
from ...trainer_plugins.distributed.accelerate import get_available_accelerator
from .constants import DeviceType, KernelType


class KernelRegistry:
    _instance: Optional['KernelRegistry'] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> 'KernelRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._registry: dict[KernelType, dict[DeviceType, Callable[..., Any]]] = {}
        self._initialized = True

    def register(
        self,
        kernel_type: KernelType,
        device_type: DeviceType,
        kernel_impl: Optional[Callable[..., Any]]
    ) -> None:
        """Register a kernel implementation.

        Args:
            kernel_type: the type of the kernel (e.g., KernelType.FLASH_ATTENTION).
            device_type: the device type the kernel is adapted to (e.g., DeviceType.CUDA).
            kernel_impl: the actual kernel function or class.
        """
        if kernel_type not in self._registry:
            self._registry[kernel_type] = {}

        if device_type in self._registry[kernel_type]:
            print(f"Warning: Overwriting kernel for {kernel_type.name} on {device_type.name}.")

        self._registry[kernel_type][device_type] = kernel_impl
        print(f"Registered kernel {kernel_type.name} for device {device_type.name}.")

    def get_kernel(
        self,
        kernel_type: KernelType,
        device_type: DeviceType
    ) -> Optional[Callable[..., Any]]:
        return self._registry.get(kernel_type, {}).get(device_type)


KERNEL_REGISTRY = KernelRegistry()


class MetaKernel(ABC):
    type: Optional[KernelType] = None
    device: Optional[DeviceType] = None
    kernel: Optional[Callable] = None

    @classmethod
    def register_kernel(cls, kernel_type: KernelType, device_type: DeviceType):
        KERNEL_REGISTRY.register(kernel_type, device_type, cls)

    @classmethod
    @abstractmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        raise NotImplementedError


class MetaFlashAttentionKernel(MetaKernel):

    @classmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        raise NotImplementedError


class MetaRMSNormKernel(MetaKernel):

    @classmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        raise NotImplementedError


class MetaSwiGluKernel(MetaKernel):

    @classmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        raise NotImplementedError


class MetaRoPEKernel(MetaKernel):

    @classmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        raise NotImplementedError


class MetaMoEKernel(MetaKernel):

    @classmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        raise NotImplementedError


def discover_kernels(model: HFModel) -> list[MetaKernel]:
    """Discover and construct MetaKernel instances for the current model/device.

    This is a placeholder to be implemented: it should inspect the runtime
    environment (device type, available extensions, model architecture) and
    return an ordered list of MetaKernel instances to be applied. Each returned
    MetaKernel must encapsulate its own replacement logic in `apply`.
    """
    # TODO: Implement auto discovery logic based on registry and device capabilities.
    return []


def apply_kernel(model: HFModel, kernel: type[MetaKernel], /, **kwargs) -> 'HFModel':
    """Call the MetaKernel's `apply` to perform the replacement.

    Corresponding replacement logic is maintained inside each kernel; the only
    requirement is that `apply` returns the replaced model.

    Example:
        from transformers import AutoModelForCausalLM
        from .rms_norm.npu_rms_norm import NpuRMSNormKernel
        model = AutoModelForCausalLM.from_pretrained("qwen/qwen2.5-0.5B")
        model = apply_kernel(model, NpuRMSNormKernel)
    """
    if issubclass(kernel, MetaKernel) and kernel.device == get_available_accelerator().type:
        return kernel.apply(model, **kwargs)

    raise ValueError(f"{kernel} must be a MetaKernel instance, or the kernel don't match the device type. got {kernel.device} and {get_available_accelerator().type} instead.")
