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

from ....utils.logging import get_logger
from ....utils.plugin import BasePlugin
from .registry import Registry


logger = get_logger(__name__)


def scan_all_kernels():
    r"""Scan all kernels in the ``ops`` directory.

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
    r"""Get a list of default registered kernel IDs.

    Returns:
        list[str]: List of kernel IDs.
    """
    return list(default_kernels.keys())


def apply_kernel(kernel_id: str, **kwargs):
    r"""Applies a specific kernel to the model.

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
    r"""Plugin for managing kernel optimizations."""

    pass


@KernelPlugin("auto").register
def apply_default_kernels(**kwargs):
    r"""Applies all default registered kernels to the model.

    Args:
        **kwargs: Keyword arguments passed to the kernel application function.
                  Typically includes the model instance and the include_kernels configuration.

    Returns:
        HFModel: The model with applied kernels.
    """
    if not kwargs.get("include_kernels"):  # None/False/empty string
        return kwargs.get("model")
    elif kwargs.get("include_kernels") == "auto" or kwargs.get("include_kernels") is True:  # True/auto
        use_kernels = default_kernels.keys()
    else:
        use_kernels = kwargs.get("include_kernels").split(",")  # "kernel_id1,kernel_id2,kernel_id3"
    for kernel in use_kernels:
        if kernel not in default_kernels:
            raise ValueError(f"Kernel {kernel} not found")
        apply_kernel(kernel, **kwargs)
    return kwargs.get("model")
