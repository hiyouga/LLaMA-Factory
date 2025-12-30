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

"""The definition of NPU fused SwiGLU kernels.

Init Phase:
1. Define SwiGLU forward functions.
2. Register NPU fused SwiGLU kernel.

"""

import re
import types

import torch

from ......accelerator.helper import DeviceType
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel


try:
    import torch_npu
except ImportError:
    pass


def npu_swiglu_forward(self, hidden_state):
    r"""SwiGLU forward pass for NPU.

    Args:
        self: The MLP layer instance.
        hidden_state (Tensor): Input hidden state.

    Returns:
        Tensor: Output of SwiGLU.
    """
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )


def _npu_swiglu_glm4_forward(self, hidden_states):
    r"""SwiGLU forward pass for GLM4 on NPU.

    Args:
        self: The GLM4 MLP layer instance.
        hidden_states (Tensor): Input hidden states.

    Returns:
        Tensor: Output of SwiGLU.
    """
    up_states = self.gate_up_proj(hidden_states)
    gate, up_states = up_states.chunk(2, dim=-1)
    return self.down_proj(torch_npu.npu_swiglu(torch.cat((gate, up_states), dim=-1), dim=-1))


def _npu_swiglu_gemma3ntext_forward(self, hidden_states):
    r"""SwiGLU forward pass for Gemma3nText on NPU.

    Args:
        self: The Gemma3nText MLP layer instance.
        hidden_states (Tensor): Input hidden states.

    Returns:
        Tensor: Output of SwiGLU.
    """
    gate_proj = self.gate_proj(hidden_states)
    if self.activation_sparsity > 0.0:
        gate_proj = self._gaussian_topk(gate_proj)
    down_proj = self.down_proj(
        torch_npu.npu_swiglu(torch.cat((gate_proj, self.up_proj(hidden_states)), dim=-1), dim=-1)
    )
    return down_proj


@register_kernel
class NpuSwiGluKernel(BaseKernel):
    r"""NPU Kernel for fused SwiGLU activation."""

    # just support apply to the following module layers
    expect_modules = frozenset(
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

    _kernel_id = "npu_fused_swiglu"
    _device = DeviceType.NPU

    @classmethod
    def apply(cls, **kwargs) -> "HFModel":
        r"""Applies the NPU fused SwiGLU kernel to the model.

        Args:
            **kwargs: Keyword arguments containing the model.

        Returns:
            HFModel: The model with patched SwiGLU forward functions.

        Raises:
            ValueError: If the model is not provided.
            RuntimeError: If dependencies are not met.
        """
        model = kwargs.get("model", None)
        if model is None:
            raise ValueError(f"HFModel instance is required for {cls.__name__}.")

        if not cls.check_deps():
            raise RuntimeError("torch_npu is not available but NpuSwiGluKernel was called.")

        # Mapping of specific mlp modules to their corresponding kernel implementations
        kernel_mapping = {
            "Glm4MLP": _npu_swiglu_glm4_forward,
            "Glm4vTextMLP": _npu_swiglu_glm4_forward,
            "Phi3MLP": _npu_swiglu_glm4_forward,
            "Gemma3nTextMLP": _npu_swiglu_gemma3ntext_forward,
        }

        swiglu_pattern = re.compile("MLP", re.IGNORECASE)
        for name, module in model.named_modules():
            # Match any module whose class name contains "MLP"
            if (
                re.search(swiglu_pattern, module.__class__.__name__)
                and module.__class__.__name__ in cls.expect_modules
            ):
                # Bind function as an instance method to preserve `self` semantics
                # and replace the original forward
                kernel_func = kernel_mapping.get(module.__class__.__name__, npu_swiglu_forward)
                module.forward = types.MethodType(kernel_func, module)

        return model
