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
import torch_npu

from .....extras.types import HFModel
from ....trainer_plugins.distributed.accelerate import is_torch_npu_available
from ..constants import DeviceType, KernelType
from ..registry import MetaMoEKernel


class GmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        fwd_output = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=1
        )[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list

        weight = torch.transpose(weight, 1, 2)
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=1
        )[0]
        grad_weight = torch_npu.npu_grouped_matmul(
            [input_tensor.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        return grad_input, grad_weight, None


def npu_group_gemm(x, weight, group_list):
    output = GmmFunction.apply(x, weight, group_list)
    return output


def npu_experts_qwen3vlmoe_forward(
    self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states, router_indices.to(torch.int32)
    )
    tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
    intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, self.gate_up_proj, tokens_per_expert)
    intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
    output = npu_group_gemm(intermediate_activations, self.down_proj, tokens_per_expert)
    next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
    next_states = next_states.view(batch_size, -1, self.hidden_size)
    return next_states


def npu_moe_block_qwen3vlmoe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    router_logits = self.gate(hidden_states)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
    routed_out = self.experts(hidden_states, routing_weights, router_indices)
    return routed_out


class NpuQwen3VLMoEFusedMoEKernel(MetaMoEKernel):
    type = KernelType.MOE
    device = DeviceType.NPU
    npu_experts_kernel = npu_experts_qwen3vlmoe_forward
    npu_moe_block_kernel = npu_moe_block_qwen3vlmoe_forward

    @classmethod
    def apply(cls, model, **kwargs) -> HFModel:
        if not is_torch_npu_available():
            return model

        npu_experts_pattern = re.compile("Qwen3VLMoeTextExperts", re.IGNORECASE)
        npu_moe_block_pattern = re.compile("Qwen3VLMoeTextSparseMoeBlock", re.IGNORECASE)

        for _, module in model.named_modules():
            if re.search(npu_experts_pattern, module.__class__.__name__):
                module.forward = types.MethodType(cls.npu_experts_kernel, module)
            elif re.search(npu_moe_block_pattern, module.__class__.__name__):
                module.forward = types.MethodType(cls.npu_moe_block_kernel, module)
        return model
