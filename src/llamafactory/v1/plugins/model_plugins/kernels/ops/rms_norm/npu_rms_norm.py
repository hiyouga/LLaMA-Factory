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

"""The definition of NPU fused RMSNorm kernels.

Init Phase:
1. Define RMSNorm forward function.
2. Register NPU fused RMSNorm kernel.

"""

import re
import types

from ......accelerator.helper import DeviceType
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel


def npu_rms_norm_forward(self, hidden_states):
    r"""NPU forward implementation for RMSNorm.

    Args:
        self: RMSNorm module instance with `weight` and `variance_epsilon`.
        hidden_states (Tensor): Input hidden states tensor, same shape as the baseline.

    Returns:
        Tensor: Normalized tensor consistent with the baseline RMSNorm behavior.
    """
    import torch_npu

    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]


@register_kernel
class NpuRMSNormKernel(BaseKernel):
    r"""NPU kernel wrapper for RMSNorm that applies the replacement within a model."""

    _kernel_id = "npu_fused_rmsnorm"
    _device = DeviceType.NPU

    @classmethod
    def apply(cls, **kwargs) -> "HFModel":
        r"""Iterate the model and apply NPU-optimized forward to matched RMSNorm modules.

        Key points:
        - Match modules whose class name contains "RMSNorm" (case-insensitive).
        - Bind `_npu_rms_forward` as an instance method via `types.MethodType` to
          replace the original `forward`.
        - Do not modify weights, hyperparameters, or module structure to ensure
          numerical behavior and interface consistency.

        Args:
            **kwargs: Keyword arguments containing the model.

        Returns:
            HFModel: The model with NPU fused RMSNorm.

        Raises:
            RuntimeError: If torch_npu is not available.
            ValueError: If the model is not provided.
        """
        model = kwargs.get("model")
        if model is None:
            raise ValueError(f"HFModel instance is required for {cls.__name__}.")

        if not cls.check_deps():
            raise RuntimeError(f"torch_npu is not available but {cls.__name__} was called.")
        rms_norm_pattern = re.compile("RMSNorm", re.IGNORECASE)

        for name, module in model.named_modules():
            # Match any module whose class name contains "RMSNorm"
            if re.search(rms_norm_pattern, module.__class__.__name__):
                # Bind function as an instance method to preserve `self` semantics
                # and replace the original forward
                module.forward = types.MethodType(npu_rms_norm_forward, module)

        return model
