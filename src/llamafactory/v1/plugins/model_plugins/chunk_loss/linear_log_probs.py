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

import torch
import torch.nn.functional as F
from torch import Tensor


class _ChunkedLinearLogProbsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        head_weight: Tensor,
        head_bias: Tensor | None,
        labels: Tensor,
        chunk_size: int,
    ) -> tuple[Tensor, Tensor]:
        per_token_log_probs = torch.empty(labels.shape, device=hidden_states.device, dtype=torch.float32)
        per_token_logits_mean = torch.empty_like(per_token_log_probs)
        log_prob_chunks = torch.split(per_token_log_probs, chunk_size, dim=1)
        logits_mean_chunks = torch.split(per_token_logits_mean, chunk_size, dim=1)
        hidden_chunks = torch.split(hidden_states, chunk_size, dim=1)
        label_chunks = torch.split(labels, chunk_size, dim=1)

        for log_prob_chunk, logits_mean_chunk, hidden_chunk, label_chunk in zip(
            log_prob_chunks,
            logits_mean_chunks,
            hidden_chunks,
            label_chunks,
            strict=True,
        ):
            logits = F.linear(hidden_chunk, head_weight, head_bias).float()
            chunk_log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                label_chunk.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(label_chunk)
            log_prob_chunk.copy_(chunk_log_probs)
            logits_mean_chunk.copy_(logits.mean(dim=-1))

        empty = torch.empty(0, device=hidden_states.device)
        ctx.save_for_backward(
            hidden_states,
            head_weight,
            head_bias if head_bias is not None else empty,
            labels,
        )
        ctx.has_bias = head_bias is not None
        ctx.chunk_size = chunk_size
        ctx.mark_non_differentiable(per_token_logits_mean)
        return per_token_log_probs, per_token_logits_mean

    @staticmethod
    def backward(ctx, grad_output: Tensor, _grad_logits_mean: Tensor):
        hidden_states, head_weight, saved_bias, labels = ctx.saved_tensors
        head_bias = saved_bias if ctx.has_bias else None
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]

        grad_hidden = torch.empty_like(hidden_states) if needs_hidden_grad else None
        grad_weight = torch.zeros_like(head_weight) if needs_weight_grad else None
        grad_bias = torch.zeros_like(head_bias) if head_bias is not None and needs_bias_grad else None

        hidden_chunks = torch.split(hidden_states, ctx.chunk_size, dim=1)
        label_chunks = torch.split(labels, ctx.chunk_size, dim=1)
        grad_output_chunks = torch.split(grad_output, ctx.chunk_size, dim=1)
        grad_hidden_chunks = torch.split(grad_hidden, ctx.chunk_size, dim=1) if grad_hidden is not None else None

        for index, (hidden_chunk, label_chunk, grad_output_chunk) in enumerate(
            zip(hidden_chunks, label_chunks, grad_output_chunks, strict=True)
        ):
            with torch.enable_grad():
                hidden_arg = hidden_chunk.detach().requires_grad_(needs_hidden_grad)
                weight_arg = head_weight.detach().requires_grad_(needs_weight_grad)
                bias_arg = None
                if head_bias is not None:
                    bias_arg = head_bias.detach().requires_grad_(needs_bias_grad)

                logits = F.linear(hidden_arg, weight_arg, bias_arg).float()
                chunk_log_probs = -F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    label_chunk.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(label_chunk)

                grad_targets = []
                if needs_hidden_grad:
                    grad_targets.append(hidden_arg)
                if needs_weight_grad:
                    grad_targets.append(weight_arg)
                if bias_arg is not None and needs_bias_grad:
                    grad_targets.append(bias_arg)
                chunk_grads = (
                    torch.autograd.grad(chunk_log_probs, grad_targets, grad_outputs=grad_output_chunk)
                    if grad_targets
                    else []
                )

            grad_index = 0
            if grad_hidden_chunks is not None:
                grad_hidden_chunks[index].copy_(chunk_grads[grad_index])
                grad_index += 1
            if grad_weight is not None:
                grad_weight.add_(chunk_grads[grad_index])
                grad_index += 1
            if grad_bias is not None:
                grad_bias.add_(chunk_grads[grad_index])

        return grad_hidden, grad_weight, grad_bias, None, None


def chunked_linear_log_probs(
    hidden_states: Tensor,
    head_weight: Tensor,
    head_bias: Tensor | None,
    labels: Tensor,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Compute target-token log-probabilities and per-token mean logits without full-sequence logits."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("Chunked linear log-probs expects hidden_states and labels to match in batch/sequence shape.")

    return _ChunkedLinearLogProbsFunction.apply(
        hidden_states,
        head_weight,
        head_bias,
        labels,
        chunk_size,
    )
