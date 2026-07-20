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

"""The definition of kernel interface.

Init Phase:
1. Scan all kernels.
2. Register default kernels.
3. Define kernel plugin.

"""

import importlib
from pathlib import Path
from typing import Literal

from ....utils import logging
from ....utils.plugin import BasePlugin
from ....utils.types import HFModel
from .registry import Registry


logger = logging.get_logger(__name__)


def scan_all_kernels():
    """Scan all kernels in the ``ops`` directory.

    Scans the ``ops`` directory for all ``.py`` files and attempts to import them.
    Importing triggers the :func:`~registry.register_kernel` decorator, which automatically registers the kernels.

    Returns:
        dict[str, type[BaseKernel]]: A dictionary of registered kernels.

    .. note::
        This function assumes that the ``ops`` directory is located in the same directory as this file.
        It recursively searches for ``.py`` files and constructs the module path for import.
    """
    ops_path = Path(__file__).parent / "ops"

    if not ops_path.exists():
        return

    base_package = __package__

    for file_path in ops_path.rglob("*.py"):
        if file_path.name == "__init__.py":
            continue

        # calculate the relative path:
        # file_path = .../kernels_v2/ops/mlp/npu_swiglu.py
        # rel_path  = ops/mlp/npu_swiglu.py
        rel_path = file_path.relative_to(Path(__file__).parent)

        # build module path:
        module_name = ".".join(rel_path.parts)[:-3]
        full_module_name = f"{base_package}.{module_name}"

        try:
            importlib.import_module(full_module_name)
        except Exception as e:
            logger.warning(f"[Kernel Registry] Failed to import {full_module_name} when loading kernels: {e}")

    return Registry.get_registered_kernels()


default_kernels = scan_all_kernels()


def get_default_kernels():
    """Get a list of default registered kernel IDs.

    Returns:
        list[str]: List of kernel IDs.
    """
    return list(default_kernels.keys())


def apply_kernel(kernel_id: str, **kwargs):
    """Applies a specific kernel to the model.

    Args:
        kernel_id (str): The ID of the kernel to apply.
        **kwargs: Keyword arguments passed to the kernel application function.
                  Typically includes the model instance.

    Returns:
        HFModel: The model with applied kernel.
    """
    kernel = default_kernels.get(kernel_id)
    if kernel is None:
        raise ValueError(f"Kernel {kernel_id} not found")

    kernel.apply(**kwargs)


class KernelPlugin(BasePlugin):
    """Plugin for managing kernel optimizations."""

    pass


@KernelPlugin("auto").register()
def apply_default_kernels(model: HFModel, include_kernels: str = None) -> HFModel:
    """Applies all default registered kernels to the model.

    Args:
        model (HFModel): The model instance to apply kernels to.
        include_kernels (str, optional): Comma-separated list of kernel IDs to apply.
                                         If "auto" or True, applies all default kernels.
                                         If None or False, no kernels are applied.
                                         Defaults to None.

    Returns:
        HFModel: The model with applied kernels.
    """
    if not include_kernels:
        return model
    elif include_kernels == "auto" or include_kernels is True:
        use_kernels = default_kernels.keys()
    else:
        use_kernels = [kernel.strip() for kernel in include_kernels.split(",") if kernel.strip()]

    for kernel in use_kernels:
        if kernel not in default_kernels:
            raise ValueError(f"Kernel {kernel} not found")

        apply_kernel(kernel, model=model)

    return model


@KernelPlugin("liger_kernel").register()
def apply_liger_kernels(
    model: HFModel,
    include_kernels: str = None,
    require_logits: bool = False,
) -> HFModel:
    """Applies Liger kernel to the model.

    Args:
        model (HFModel): The model instance to apply kernels to.
        include_kernels (str, optional): If ``"auto"`` or ``True``, apply Liger with
                                         library defaults. If a comma-separated list (e.g.
                                         ``rope,rms_norm``), enable only those ops; names match
                                         ``apply_liger_kernel_to_*`` kwargs: ``rope``, ``rms_norm``,
                                         ``swiglu``, ``cross_entropy``, ``fused_linear_cross_entropy``.
                                         If ``None`` or ``False``, do nothing. Defaults to ``None``.
        require_logits (bool, optional): When true, disables ``fused_linear_cross_entropy`` in favor
                                         of non-fused CE so the forward pass returns ``logits``. Needed
                                         for trainers that compute weighted loss from logits (e.g. v1
                                         SFT with ``loss_weights``). Defaults to ``False`` (fused CE
                                         when supported). The v1 ``run_sft`` entrypoint sets
                                         ``require_logits`` to true for ``liger_kernel`` when the key
                                         is omitted so SFT weighted loss keeps working.

    Returns:
        HFModel: The model with Liger kernel applied.
    """
    if not include_kernels:
        return model
    if include_kernels == "auto" or include_kernels is True:
        use_kernels = "auto"
    else:
        use_kernels = [k.strip() for k in include_kernels.split(",") if k.strip()]
        if not use_kernels:
            return model

    try:
        from .liger_kernel_ops import LigerKernel
    except ImportError as e:
        logger.warning_rank0(f"[Kernel] Failed to import liger_kernel ops, skip. Error: {e}")
        return model

    return LigerKernel.apply(use_kernels=use_kernels, model=model, require_logits=require_logits)


@KernelPlugin("flash-linear-attention").register()
def apply_flash_linear_attention_kernels(
    model: HFModel,
    include_kernels: str = None,
    chunk_size: Literal[16, 32, 64] = 64,
) -> HFModel:
    """Apply selected Flash Linear Attention kernels through the standard kernel registry."""
    if not include_kernels:
        return model

    from .ops.linear_attention.fla import (
        CHUNK_GATED_DELTA_RULE,
        FLASH_LINEAR_ATTENTION_KERNELS,
        SUPPORTED_CHUNK_SIZES,
    )

    if type(chunk_size) is not int or chunk_size not in SUPPORTED_CHUNK_SIZES:
        raise ValueError(f"chunk_size must be one of {SUPPORTED_CHUNK_SIZES}, got {chunk_size!r}.")

    if include_kernels == "auto" or include_kernels is True:
        use_kernels = FLASH_LINEAR_ATTENTION_KERNELS
    else:
        use_kernels = [name.strip() for name in include_kernels.split(",") if name.strip()]

    unsupported = set(use_kernels).difference(FLASH_LINEAR_ATTENTION_KERNELS)
    if unsupported:
        raise ValueError(f"Unsupported Flash Linear Attention kernels: {sorted(unsupported)}")

    for kernel in use_kernels:
        kernel_kwargs = {"model": model, "strict": True}
        if kernel == CHUNK_GATED_DELTA_RULE:
            kernel_kwargs["op_kwargs"] = {"chunk_size": chunk_size}
        apply_kernel(kernel, **kernel_kwargs)

    return model
