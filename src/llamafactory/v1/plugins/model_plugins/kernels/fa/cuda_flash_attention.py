import importlib.util

from .....extras.types import DeviceType, HFModel, KernelType
from ....trainer_plugins.distributed.accelerate import is_torch_cuda_available
from ..registry import KERNEL_REGISTRY, MetaFlashAttentionKernel


class CUDAFlashAttention2Kernel(MetaFlashAttentionKernel):

    device = DeviceType.CUDA
    kernel = None

    @classmethod
    def register_kernel(cls, kernel_type=KernelType.FLASH_ATTENTION, device_type=device):
        KERNEL_REGISTRY.register(kernel_type, device_type, cls)

    @classmethod
    def apply(cls, model, **kwargs) -> HFModel:
        if is_torch_cuda_available() and importlib.util.find_spec("flash_attn"):
            model.set_attn_implementation('flash_attention_2')
        return model


class CUDAFlashAttention3Kernel(MetaFlashAttentionKernel):

    device = DeviceType.CUDA
    kernel = None

    @classmethod
    def register_kernel(cls, kernel_type=KernelType.FLASH_ATTENTION, device_type=device):
        KERNEL_REGISTRY.register(kernel_type, device_type, cls)

    @classmethod
    def apply(cls, model, **kwargs) -> HFModel:
        if is_torch_cuda_available() and importlib.util.find_spec("flash_attn"):
            model.set_attn_implementation('flash_attention_3')
        return model
