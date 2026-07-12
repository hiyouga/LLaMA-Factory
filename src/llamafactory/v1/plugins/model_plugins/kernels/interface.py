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

from typing import Any

from ....utils import logging
from ....utils.plugin import BasePlugin
from ....utils.types import HFModel


logger = logging.get_logger(__name__)


class KernelPlugin(BasePlugin):
    """Plugin family for model kernel optimizations."""


@KernelPlugin("npu_fused_rmsnorm").register()
def apply_npu_fused_rmsnorm(model: HFModel, **kwargs) -> HFModel:
    from .ops.rms_norm.npu_rms_norm import NpuRMSNormKernel

    return NpuRMSNormKernel.apply(model=model, **kwargs)


@KernelPlugin("npu_fused_rope").register()
def apply_npu_fused_rope(model: HFModel, **kwargs) -> HFModel:
    from .ops.rope.npu_rope import NpuRoPEKernel

    return NpuRoPEKernel.apply(model=model, **kwargs)


@KernelPlugin("npu_fused_swiglu").register()
def apply_npu_fused_swiglu(model: HFModel, **kwargs) -> HFModel:
    from .ops.mlp.npu_swiglu import NpuSwiGluKernel

    return NpuSwiGluKernel.apply(model=model, **kwargs)


@KernelPlugin("npu_fused_moe").register()
def apply_npu_fused_moe(model: HFModel, **kwargs) -> HFModel:
    from .ops.mlp.npu_fused_moe import NpuFusedMoEKernel

    return NpuFusedMoEKernel.apply(model=model, **kwargs)


@KernelPlugin("cuda_fused_moe").register()
def apply_cuda_fused_moe(model: HFModel, **kwargs) -> HFModel:
    from .ops.mlp.cuda_fused_moe import CudaFusedMoEKernel

    return CudaFusedMoEKernel.apply(model=model, **kwargs)


@KernelPlugin("liger_kernel").register()
def apply_liger_kernels(model: HFModel, config: dict[str, Any] | None = None, **kwargs) -> HFModel:
    try:
        from .liger_kernel_ops import LigerKernel
    except ImportError as exc:
        logger.warning_rank0(f"[Kernel] Failed to import liger_kernel ops, skip. Error: {exc}")
        return model

    require_logits = kwargs.get("require_logits", False)
    if config is not None:
        require_logits = config.get("require_logits", require_logits)
    return LigerKernel.apply(use_kernels="auto", model=model, require_logits=require_logits)


_AUTO_KERNELS = ("npu_fused_moe", "npu_fused_rmsnorm", "npu_fused_rope", "npu_fused_swiglu")


@KernelPlugin("auto").register()
def apply_auto_kernels(model: HFModel, **kwargs) -> HFModel:
    for kernel_name in _AUTO_KERNELS:
        try:
            model = KernelPlugin(kernel_name)(model=model, **kwargs)
        except RuntimeError:
            continue
    return model


def apply_kernels(model: HFModel, config: dict[str, Any], require_logits: bool = False) -> HFModel:
    """Apply the comma-separated kernel names selected by ``kernel_config.name``."""
    kernel_names = config.get("name")
    if not isinstance(kernel_names, str):
        raise TypeError("kernel_config.name must be a string.")

    names = [name.strip() for name in kernel_names.split(",") if name.strip()]
    if not names:
        raise ValueError("kernel_config.name must contain at least one kernel name.")

    for name in names:
        model = KernelPlugin(name)(model=model, config=config, require_logits=require_logits)
    return model


def apply_v1_kernels(model: HFModel, use_v1_kernels: bool) -> HFModel:
    """Apply v1 automatic kernels for the transitional v0 ``use_v1_kernels`` option."""
    if not use_v1_kernels:
        return model

    return apply_kernels(model, {"name": "auto"})


def apply_kernel(kernel_id: str, **kwargs) -> HFModel:
    return KernelPlugin(kernel_id)(**kwargs)
