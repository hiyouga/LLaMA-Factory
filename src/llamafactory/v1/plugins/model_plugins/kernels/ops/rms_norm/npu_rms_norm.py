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

"""NPU fused RMSNorm kernel."""

import re
import types

import torch
import torch.nn.functional as F

from ......accelerator.helper import DeviceType
from ......utils.types import HFModel
from ...base import BaseKernel


try:
    import torch_npu
except ImportError:
    pass


def _should_use_residual_rmsnorm(module):
    """Detect whether the module uses residual RMSNorm parameterization.

    Residual RMSNorm uses ``scale = 1.0 + weight`` where weight is initialized to 0,
    while standard RMSNorm uses ``scale = weight`` where weight is initialized to 1.
    """
    if hasattr(module, "weight") and module.weight is not None:
        weight_mean = module.weight.data.mean().item()
        if abs(weight_mean) < 0.3:
            return True

    class_name = module.__class__.__name__
    residual_patterns = ["Qwen3_5", "Qwen3_6", "Qwen4"]
    for pattern in residual_patterns:
        if pattern in class_name:
            return True

    return False


def npu_rms_norm_forward(self, hidden_states):
    """NPU forward for standard RMSNorm."""
    _eps = getattr(self, "variance_epsilon", None) or getattr(self, "eps", 1e-6)

    if hasattr(self, "weight") and self.weight is not None:
        if _should_use_residual_rmsnorm(self):
            effective_weight = 1.0 + self.weight.float()
        else:
            effective_weight = self.weight.float()
    else:
        effective_weight = None

    if effective_weight is not None:
        return torch_npu.npu_rms_norm(hidden_states, effective_weight.to(hidden_states.dtype), epsilon=_eps)[0]
    else:
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=_eps)[0]


def npu_gated_rms_norm_forward(self, hidden_states, gate=None):
    """NPU forward for Gated RMSNorm."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    _eps = getattr(self, "variance_epsilon", None) or getattr(self, "eps", 1e-6)

    hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight.float(), epsilon=_eps)[0]

    if gate is not None:
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

    return hidden_states.to(input_dtype)


class NpuRMSNormKernel(BaseKernel):
    """NPU fused RMSNorm kernel."""

    _device = DeviceType.NPU

    @classmethod
    def _apply(cls, **kwargs) -> HFModel:
        model = kwargs.get("model")
        if model is None:
            raise ValueError("HFModel instance is required.")

        rms_norm_pattern = re.compile("RMSNorm", re.IGNORECASE)
        for _, module in model.named_modules():
            if re.search(rms_norm_pattern, module.__class__.__name__):
                if "Gated" in module.__class__.__name__:
                    module.forward = types.MethodType(npu_gated_rms_norm_forward, module)
                else:
                    module.forward = types.MethodType(npu_rms_norm_forward, module)

        return model
