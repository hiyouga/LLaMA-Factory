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

"""The definition of NPU fused MoE kernels.

Init Phase:
1. Define GMM functions.
2. Define NPU fused MoE functions.
3. Register NPU fused MoE kernel.

"""

import types

import torch
import torch.nn.functional as F


try:
    import torch_npu
except ImportError:
    pass

from ......accelerator.helper import DeviceType
from ......utils.packages import is_transformers_version_greater_than
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel


class GmmFunction(torch.autograd.Function):
    r"""Custom autograd function for NPU Grouped Matrix Multiplication (GMM)."""

    @staticmethod
    def forward(ctx, x, weight, group_list):
        r"""Performs the forward pass of Grouped Matrix Multiplication.

        Args:
            ctx: Context object to save tensors for backward pass.
            x (Tensor): Input tensor.
            weight (Tensor): Weight tensor.
            group_list (list): List of group sizes.

        Returns:
            Tensor: The result of the grouped matrix multiplication.
        """
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        fwd_output = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=1
        )[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        r"""Performs the backward pass of Grouped Matrix Multiplication.

        Args:
            ctx: Context object containing saved tensors.
            grad_output (Tensor): Gradient with respect to the output.

        Returns:
            tuple: Gradients with respect to input, weight, and None for group_list.
        """
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


class HybridGmmFunction(torch.autograd.Function):
    r"""Custom autograd function for Hybrid Grouped Matrix Multiplication on NPU."""

    @staticmethod
    def forward(ctx, num_experts, *args):
        r"""Performs the forward pass of Hybrid GMM.

        Args:
            ctx: Context object to save tensors.
            num_experts (int): Number of experts.
            *args: Variable length argument list containing inputs and weights.

        Returns:
            tuple: The outputs of the grouped matrix multiplication.
        """
        x_list = list(args[:num_experts])
        weight_list = list(args[num_experts:])

        split_sizes = [x.shape[0] for x in x_list]
        ctx.split_sizes = split_sizes
        ctx.num_experts = num_experts

        ctx.save_for_backward(*args)

        outputs = torch_npu.npu_grouped_matmul(
            x_list, weight_list, bias=None, group_list=None, split_item=0, group_type=-1
        )
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        r"""Performs the backward pass of Hybrid GMM.

        Args:
            ctx: Context object containing saved tensors.
            *grad_outputs: Gradients with respect to the outputs.

        Returns:
            tuple: Gradients with respect to inputs and weights.
        """
        saved_tensors = ctx.saved_tensors
        num_experts = ctx.num_experts
        split_sizes = ctx.split_sizes

        x_list = list(saved_tensors[:num_experts])
        weight_list = list(saved_tensors[num_experts:])

        grad_outputs_contiguous = [g.contiguous() for g in grad_outputs]

        w_t_list = [w.t() for w in weight_list]
        grad_x_list = torch_npu.npu_grouped_matmul(
            grad_outputs_contiguous,  # List[Tensor], 每个 [M_i, N]
            w_t_list,  # List[Tensor], 每个 [N, K] (view)
            bias=None,
            group_list=None,
            split_item=0,
            group_type=-1,
        )

        x_concat = torch.cat(x_list, dim=0)
        dy_concat = torch.cat(grad_outputs_contiguous, dim=0)  # [Total_M, N]

        group_list = torch.tensor(split_sizes, device=x_concat.device, dtype=torch.int64)

        grad_w_stack = torch_npu.npu_grouped_matmul(
            [x_concat.t()],
            [dy_concat],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]

        if grad_w_stack.dim() == 3:
            grad_w_list = list(torch.unbind(grad_w_stack, dim=0))
        else:
            raise RuntimeError(f"Unexpected grad_w_stack shape: {grad_w_stack.shape}")

        return (None, *grad_x_list, *grad_w_list)


class NpuMoeFused:
    r"""Container for NPU fused MoE forward functions."""

    @staticmethod
    def npu_moe_experts_forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        r"""Forward pass for MoE experts using NPU fused operations.

        Args:
            self: The MoE layer instance.
            hidden_states (Tensor): Input hidden states.
            routing_weights (Tensor): Routing weights.
            router_indices (Tensor): Router indices.

        Returns:
            Tensor: Output tensor after expert computation.
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
            hidden_states, router_indices.to(torch.int32)
        )
        tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
        intermediate_hidden_states = GmmFunction.apply(permuted_hidden_states, self.gate_up_proj, tokens_per_expert)
        intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
        output = GmmFunction.apply(intermediate_activations, self.down_proj, tokens_per_expert)
        next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states

    @staticmethod
    def npu_moe_sparse_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Forward pass for sparse MoE block using NPU optimization.

        Args:
            self: The MoE sparse block instance.
            hidden_states (Tensor): Input hidden states.

        Returns:
            Tensor: The routed output.
        """
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


class Qwen3NpuMoeFused:
    r"""Container for Qwen3 NPU fused MoE forward functions."""

    @staticmethod
    def qwen3moe_sparse_moe_block_forward(self, hidden_states: torch.Tensor):
        r"""Forward pass for Qwen3 sparse MoE block using NPU fused operations.

        Args:
            self: The Qwen3 MoE block instance.
            hidden_states (Tensor): Input hidden states.

        Returns:
            tuple: A tuple containing the next states and router logits.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.int())

        tokens_per_expert = torch.histc(
            selected_experts.float(), bins=self.num_experts, min=0, max=self.num_experts
        ).long()
        split_sizes = tokens_per_expert.tolist()

        input_list = list(torch.split(permuted_hidden_states, split_sizes, dim=0))

        gate_weights = [e.gate_proj.weight.t() for e in self.experts]
        up_weights = [e.up_proj.weight.t() for e in self.experts]
        down_weights = [e.down_proj.weight.t() for e in self.experts]

        gate_out_tuple = HybridGmmFunction.apply(len(input_list), *input_list, *gate_weights)
        up_out_tuple = HybridGmmFunction.apply(len(input_list), *input_list, *up_weights)

        inter_list = [F.silu(g) * u for g, u in zip(gate_out_tuple, up_out_tuple)]

        down_out_tuple = HybridGmmFunction.apply(len(inter_list), *inter_list, *down_weights)

        grouped_output = torch.cat(down_out_tuple, dim=0)

        next_states = torch_npu.npu_moe_token_unpermute(grouped_output, row_ids_map, probs=routing_weights)

        next_states = next_states.view(batch_size, sequence_length, -1)
        return next_states, router_logits


# moe patch config mapping
kernel_moe_mapping = {
    "Qwen3VLMoeForConditionalGeneration": {
        "Qwen3VLMoeTextExperts": NpuMoeFused.npu_moe_experts_forward,
        "Qwen3VLMoeTextSparseMoeBlock": NpuMoeFused.npu_moe_sparse_block_forward,
    }
}

if not is_transformers_version_greater_than("5.0.0"):
    kernel_moe_mapping["Qwen3MoeForCausalLM"] = {
        "Qwen3MoeSparseMoeBlock": Qwen3NpuMoeFused.qwen3moe_sparse_moe_block_forward
    }


@register_kernel
class NpuFusedMoEKernel(BaseKernel):
    r"""NPU Fused MoE Kernel implementation."""

    _kernel_id = "npu_fused_moe"
    _device = DeviceType.NPU

    @classmethod
    def apply(cls, **kwargs) -> HFModel:
        r"""Applies the NPU fused MoE kernel to the model.

        Args:
            **kwargs: Keyword arguments containing the model.

        Returns:
            HFModel: The model with patched MoE forward functions.

        Raises:
            ValueError: If the model is not provided.
            RuntimeError: If dependencies are not met.
        """
        model = kwargs.get("model", None)
        if model is None:
            raise ValueError(f"HFModel instance is required for {cls.__name__}.")

        if not cls.check_deps():
            raise RuntimeError("torch_npu is not available but NpuMoEFusedMoEKernel was called.")

        archs = getattr(model.config, "architectures", [])
        target_moe_mapping = None
        for arch in archs:
            if arch in kernel_moe_mapping:
                target_moe_mapping = kernel_moe_mapping[arch]
                break

        if target_moe_mapping is None:
            return model
        for module in model.modules():
            class_name = module.__class__.__name__
            if class_name in target_moe_mapping:
                new_forward_func = target_moe_mapping[class_name]
                module.forward = types.MethodType(new_forward_func, module)

        return model
