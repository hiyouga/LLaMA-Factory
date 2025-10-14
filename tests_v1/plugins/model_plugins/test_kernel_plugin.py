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

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * sin
    k_embed = k * cos
    return q_embed, k_embed


class TinyRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(10, 10)
        self.up_proj = nn.Linear(10, 10)
        self.down_proj = nn.Linear(10, 10)

    def forward(self, x):
        return self.gate_proj(x) * self.up_proj(x) + self.down_proj(x)


class TinyAttention(nn.Module):
    def forward(self, q, k, v, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)
        return q_embed, k_embed


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.norm = TinyRMSNorm(10)
        self.mlp = TinyMLP()
        self.attn = TinyAttention()
        self.attn_implementation = 'default'

    def set_attn_implementation(self, attn_implementation):
        self.attn_implementation = attn_implementation

    def forward(self, x):
        return self.mlp(self.norm(self.linear(x)))


class TestKernelPlugin(unittest.TestCase):

    @patch('torch.accelerator.current_accelerator')
    def test_apply_kernel(self, mock_get_accelerator):
        mock_device = MagicMock()
        mock_device.type = 'npu'
        mock_get_accelerator.return_value = mock_device

        model = TinyModel()

        original_rmsnorm_forward = model.norm.forward
        original_swiglu_forward = model.mlp.forward


        from llamafactory.v1.plugins.model_plugins.kernels.mlp import npu_swiglu
        from llamafactory.v1.plugins.model_plugins.kernels.registry import apply_kernel
        from llamafactory.v1.plugins.model_plugins.kernels.rms_norm import npu_rms_norm
        from llamafactory.v1.plugins.model_plugins.kernels.rope import npu_rope

        apply_kernel(model, npu_rope.NpuRoPEKernel)

        model = apply_kernel(model, npu_rms_norm.NpuRMSNormKernel)
        assert model.norm.forward is not original_rmsnorm_forward

        model = apply_kernel(model, npu_swiglu.NpuSwiGluKernel)
        assert model.mlp.forward is not original_swiglu_forward
