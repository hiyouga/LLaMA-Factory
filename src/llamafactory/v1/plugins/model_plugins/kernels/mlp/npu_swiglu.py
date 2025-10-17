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

import re
import types

import torch

from .....extras.types import HFModel
from ....trainer_plugins.distributed.accelerate import is_torch_npu_available
from ..constants import DeviceType, KernelType
from ..registry import KERNEL_REGISTRY, MetaSwiGluKernel


def _npu_swiglu_forward(self, hidden_state):
    import torch_npu

    return self.down_proj(
        torch_npu.npu_swiglu(
            torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1
        )
    )


class NpuSwiGluKernel(MetaSwiGluKernel):

    device = DeviceType.NPU
    kernel = _npu_swiglu_forward

    @classmethod
    def register_kernel(cls, kernel_type=KernelType.SWIGLU, device_type=DeviceType.NPU):
        KERNEL_REGISTRY.register(kernel_type, device_type, cls)

    @classmethod
    def apply(cls, model, **kwargs) -> 'HFModel':
        if not is_torch_npu_available():
            return model

        swiglu_pattern = re.compile("MLP", re.IGNORECASE)
        for name, module in model.named_modules():
            # Match any module whose class name contains "RMSNorm"
            if re.search(swiglu_pattern, module.__class__.__name__):

                # Bind function as an instance method to preserve `self` semantics
                # and replace the original forward
                module.forward = types.MethodType(cls.kernel, module)

        return model
