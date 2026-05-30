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

"""NPU fused SwiGLU kernel."""

import re
import types

import torch

from ......accelerator.helper import DeviceType
from ......utils.types import HFModel
from ...base import BaseKernel


try:
    import torch_npu
except ImportError:
    pass


def npu_swiglu_forward(self, hidden_state):
    """SwiGLU forward pass for NPU."""
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )


def _npu_swiglu_glm4_forward(self, hidden_states):
    """SwiGLU forward pass for GLM4 on NPU."""
    up_states = self.gate_up_proj(hidden_states)
    gate, up_states = up_states.chunk(2, dim=-1)
    return self.down_proj(torch_npu.npu_swiglu(torch.cat((gate, up_states), dim=-1), dim=-1))


def _npu_swiglu_gemma3ntext_forward(self, hidden_states):
    """SwiGLU forward pass for Gemma3nText on NPU."""
    gate_proj = self.gate_proj(hidden_states)
    if self.activation_sparsity > 0.0:
        gate_proj = self._gaussian_topk(gate_proj)
    return self.down_proj(torch_npu.npu_swiglu(torch.cat((gate_proj, self.up_proj(hidden_states)), dim=-1), dim=-1))


_EXPECT_MODULES = frozenset(
    {
        "Qwen3VLMoeTextMLP",
        "Qwen3VLTextMLP",
        "Qwen3OmniMoeThinkerTextMLP",
        "Qwen3OmniMoeMLP",
        "Qwen3OmniMoeTalkerTextMLP",
        "Qwen3OmniMoeCode2WavMlp",
        "Qwen3NextMLP",
        "Qwen3MoeMLP",
        "Qwen3MLP",
        "Qwen2MLP",
        "Qwen2MoeMLP",
        "Qwen2_5_VLMLP",
        "Qwen2_5OmniMLP",
        "Llama4TextMLP",
        "LlamaMLP",
        "Glm4MLP",
        "Glm4MoeMLP",
        "Glm4vMoeTextMLP",
        "Gemma3MLP",
        "Gemma2MLP",
        "Gemma3nTextMLP",
        "Phi3MLP",
        "DeepseekV2MLP",
        "DeepseekV3MLP",
        "SeedOssMLP",
    }
)

_KERNEL_MAPPING = {
    "Glm4MLP": _npu_swiglu_glm4_forward,
    "Glm4vTextMLP": _npu_swiglu_glm4_forward,
    "Phi3MLP": _npu_swiglu_glm4_forward,
    "Gemma3nTextMLP": _npu_swiglu_gemma3ntext_forward,
}


class NpuSwiGLUKernel(BaseKernel):
    """NPU fused SwiGLU kernel."""

    _device = DeviceType.NPU

    @classmethod
    def _apply(cls, **kwargs) -> HFModel:
        model = kwargs.get("model")
        if model is None:
            raise ValueError("HFModel instance is required.")

        swiglu_pattern = re.compile("MLP", re.IGNORECASE)
        for _, module in model.named_modules():
            if re.search(swiglu_pattern, module.__class__.__name__) and module.__class__.__name__ in _EXPECT_MODULES:
                kernel_func = _KERNEL_MAPPING.get(module.__class__.__name__, npu_swiglu_forward)
                module.forward = types.MethodType(kernel_func, module)

        return model
