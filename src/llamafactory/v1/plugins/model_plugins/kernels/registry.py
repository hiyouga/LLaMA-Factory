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

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Optional, Union

from ....accelerator.helper import DeviceType, get_current_accelerator
from ....utils.types import HFModel
from .constants import KernelType


class KernelRegistry:
    _instance: Optional["KernelRegistry"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "KernelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._registry: dict[KernelType, dict[DeviceType, Callable[..., Any]]] = {}
        self._initialized = True

    def register(
        self, kernel_type: KernelType, device_type: DeviceType, kernel_impl: Optional[Callable[..., Any]]
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

    def get_kernel(self, kernel_type: KernelType, device_type: DeviceType) -> Optional[Callable[..., Any]]:
        return self._registry.get(kernel_type, {}).get(device_type)


KERNEL_REGISTRY = KernelRegistry()


class AutoRegisterKernelMeta(ABCMeta):
    """Metaclass that automatically registers kernel classes upon creation.

    This metaclass checks if a newly created class has both `type` and `device`
    attributes defined. If so, it automatically registers the kernel in the
    global KERNEL_REGISTRY, eliminating the need for manual registration.

    To disable auto-registration for a specific class, set `auto_register = False`.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Check if auto-registration is disabled
        auto_register = namespace.get("auto_register", True)

        # Only auto-register if the class has both type and device attributes defined
        # and they are not None (skip base classes like MetaKernel itself)
        # and auto_register is True
        kernel_type = namespace.get("type")
        device_type = namespace.get("device")

        if auto_register and kernel_type is not None and device_type is not None:
            # Auto-register this kernel
            KERNEL_REGISTRY.register(kernel_type, device_type, cls)

        return cls


class MetaKernel(ABC, metaclass=AutoRegisterKernelMeta):
    """Base class for all kernel implementations.

    Subclasses are automatically registered when they define both `type` and `device`
    attributes. To disable auto-registration, set `auto_register = False`.

    Attributes:
        type: The kernel type (e.g., KernelType.RMSNORM). Must be set in subclasses.
        device: The device type (e.g., DeviceType.NPU). Must be set in subclasses.
        kernel: The actual kernel function or implementation.
        auto_register: Set to False to disable automatic registration (default: True).
    """

    type: Optional[KernelType] = None
    device: Optional[DeviceType] = None
    kernel: Optional[Callable] = None

    @classmethod
    @abstractmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        """Apply the kernel to the model.

        This method should check if the kernel can be applied (e.g., dependencies
        are installed, target modules exist) and perform the kernel replacement.

        Args:
            model: The HuggingFace model to optimize.
            **kwargs: Additional arguments for kernel application.

        Returns:
            The optimized model (may be the same object with modifications).
        """
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


def _ensure_kernels_loaded() -> None:
    """Ensure all kernel implementations are imported and registered.

    This function dynamically imports all kernel implementation modules to trigger
    their auto-registration. Python's module system ensures each module is only
    executed once (cached in sys.modules), so repeated calls are safe and fast.
    """
    # List of kernel module paths to import
    kernel_modules = [
        "rms_norm.npu_rms_norm",
        "rope.npu_rope",
        "mlp.npu_swiglu",
        "mlp.npu_fused_moe",
        # Add new kernel modules here as they are created
    ]

    # Import each module to trigger kernel registration
    # Python's import system caches modules, so this is fast on subsequent calls
    for module_name in kernel_modules:
        try:
            __import__(f"{__package__}.{module_name}", fromlist=["*"])
        except ImportError:
            # Silently ignore import errors (e.g., missing dependencies like torch_npu)
            pass


def discover_kernels(model: HFModel = None) -> list[type[MetaKernel]]:
    """Discover and return all kernel classes registered for the current device.

    This function inspects the runtime environment (device type) and returns
    all MetaKernel classes registered for that device. Each kernel's `apply()`
    method is responsible for checking if it can actually be applied (e.g.,
    required dependencies are installed, target modules exist in the model).

    The function automatically discovers all kernels registered in KERNEL_REGISTRY
    without requiring manual enumeration. On first call, it dynamically imports
    all kernel implementation modules to trigger their auto-registration.

    Args:
        model: The HuggingFace model to apply kernels to.
        TODO: implement the kernel route detection logic by model structure.

    Returns:
        A list of MetaKernel classes available for the current device.
    """
    # Ensure all kernel modules are imported to trigger registration
    _ensure_kernels_loaded()

    discovered_kernels: list[type[MetaKernel]] = []

    # Detect current device type
    accelerator = get_current_accelerator()
    try:
        device_type = DeviceType(accelerator.type)
    except ValueError:
        # Unknown device type, return empty list
        return discovered_kernels

    # Skip CPU as it typically doesn't have optimized kernels
    if device_type == DeviceType.CPU:
        return discovered_kernels

    # Iterate through registry and collect all kernels for current device
    for devices in KERNEL_REGISTRY._registry.values():
        kernel_cls = devices.get(device_type)
        if kernel_cls is not None:
            discovered_kernels.append(kernel_cls)

    return discovered_kernels


def apply_kernel(model: HFModel, kernel: Union[type[MetaKernel], Any], /, **kwargs) -> "HFModel":
    """Call the MetaKernel's `apply` to perform the replacement.

    Corresponding replacement logic is maintained inside each kernel; the only
    requirement is that `apply` returns the replaced model.

    Example:
        from transformers import AutoModelForCausalLM
        from .rms_norm.npu_rms_norm import NpuRMSNormKernel
        model = AutoModelForCausalLM.from_pretrained("qwen/qwen2.5-0.5B")
        model = apply_kernel(model, NpuRMSNormKernel)
    """
    if not issubclass(kernel, MetaKernel):
        raise ValueError(f"{kernel} must be a MetaKernel instance.")

    if kernel.device != get_current_accelerator().type:
        raise ValueError(f"{kernel} must be applied to {kernel.device} device, got {get_current_accelerator().type}.")

    return kernel.apply(model, **kwargs)


def apply_available_kernels(model: HFModel, **kwargs) -> "HFModel":
    """Apply all available kernels to the model."""
    for kernel in discover_kernels(model):
        model = apply_kernel(model, kernel, **kwargs)

    return model
