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

"""Concrete kernel registrations and public helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ....utils import logging
from ....utils.plugin import BasePlugin


if TYPE_CHECKING:
    from ....utils.types import HFModel


logger = logging.get_logger(__name__)


class KernelPlugin(BasePlugin):
    """Plugin-family registry for kernel optimisations."""


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
    from .ops.mlp.npu_swiglu import NpuSwiGLUKernel

    return NpuSwiGLUKernel.apply(model=model, **kwargs)


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
    try:
        return LigerKernel.apply(use_kernels="auto", model=model, require_logits=require_logits)
    except RuntimeError as exc:
        logger.warning_rank0(f"[Kernel] Liger kernel is not available, skip. Error: {exc}")
        return model


_AUTO_KERNELS = (
    "npu_fused_moe",
    "npu_fused_rmsnorm",
    "npu_fused_rope",
    "npu_fused_swiglu",
)


@KernelPlugin("auto").register()
def apply_auto_kernels(model: HFModel, **kwargs) -> HFModel:
    for kernel_name in _AUTO_KERNELS:
        try:
            model = KernelPlugin(kernel_name)(model=model, **kwargs)
        except RuntimeError:
            continue
    return model


def get_auto_kernels() -> list[str]:
    return list(_AUTO_KERNELS)


def apply_kernels(model: HFModel, config: dict[str, Any]) -> HFModel:
    kernel_names = config["name"]
    if not isinstance(kernel_names, str):
        raise TypeError(f"kernel_config.name must be a string; got {type(kernel_names).__name__}.")

    kernel_names = [kernel_name.strip() for kernel_name in kernel_names.split(",") if kernel_name.strip()]
    if not kernel_names:
        raise ValueError("kernel_config.name must contain at least one kernel name.")

    for kernel_name in kernel_names:
        kwargs = {"config": config}
        if kernel_name == "liger_kernel" and "require_logits" not in config:
            kwargs["require_logits"] = True
        model = KernelPlugin(kernel_name)(model=model, **kwargs)

    return model


def apply_kernel(kernel_id: str, **kwargs) -> HFModel:
    return KernelPlugin(kernel_id)(**kwargs)
