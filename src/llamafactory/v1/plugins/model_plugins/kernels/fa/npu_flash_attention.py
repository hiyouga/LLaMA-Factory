from .....extras.types import DeviceType, HFModel, KernelType
from ....trainer_plugins.distributed.accelerate import is_torch_npu_available
from ..registry import KERNEL_REGISTRY, MetaFlashAttentionKernel


class NpuFlashAttentionKernel(MetaFlashAttentionKernel):

    device = DeviceType.NPU
    kernel = None

    def register_kernel(self, kernel_type=KernelType.FLASH_ATTENTION, device_type=device):
        KERNEL_REGISTRY.register(kernel_type, device_type, self.__class__)

    @classmethod
    def apply(cls, model, **kwargs) -> HFModel:
        if is_torch_npu_available():
            try:
                model.set_attn_implementation('kernels-ext-npu/flash-attn2')
            except Exception:
                model.set_attn_implementation('sdpa')
        return model


