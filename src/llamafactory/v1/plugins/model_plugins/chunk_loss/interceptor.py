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

from __future__ import annotations

import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from ....utils.types import HFModel
from .linear_cross_entropy import chunked_linear_cross_entropy
from .linear_log_probs import chunked_linear_log_probs
from .policy import ChunkSizePolicy


@dataclass
class _OutputHeadCallState:
    name: str
    compute_output: Callable[[nn.Linear, Tensor], Tensor]
    output_head_call_count: int = 0


_ACTIVE_OUTPUT_HEAD_CALL: ContextVar[_OutputHeadCallState | None] = ContextVar(
    "active_chunk_loss_output_head_call",
    default=None,
)


@contextmanager
def _activate_output_head_call(call_state: _OutputHeadCallState) -> Iterator[None]:
    if _ACTIVE_OUTPUT_HEAD_CALL.get() is not None:
        raise RuntimeError("Nested Chunk Loss output-head contexts are not supported.")

    token = _ACTIVE_OUTPUT_HEAD_CALL.set(call_state)
    try:
        yield
    finally:
        _ACTIVE_OUTPUT_HEAD_CALL.reset(token)


def _to_local_tensor(tensor: Tensor | None, parameter_name: str) -> Tensor | None:
    if tensor is None:
        return None

    try:
        from torch.distributed.tensor import DTensor, Replicate
    except ImportError:
        return tensor

    if not isinstance(tensor, DTensor):
        return tensor
    if not all(isinstance(placement, Replicate) for placement in tensor.placements):
        raise NotImplementedError(
            f"Chunk Loss requires an unsharded {parameter_name} DTensor during lm_head forward, "
            f"but found placements {tensor.placements}. Vocab/tensor-parallel output heads require "
            "a distributed softmax implementation."
        )

    local_tensor = tensor.to_local()
    if local_tensor.shape != tensor.shape:
        raise RuntimeError(
            f"The local {parameter_name} shape {tuple(local_tensor.shape)} does not match its global DTensor "
            f"shape {tuple(tensor.shape)}."
        )

    return local_tensor


def _get_local_output_head_parameters(output_head: nn.Linear) -> tuple[Tensor, Tensor | None]:
    weight = _to_local_tensor(output_head.weight, "lm_head.weight")
    bias = _to_local_tensor(output_head.bias, "lm_head.bias")
    assert weight is not None
    return weight, bias


def _get_linear_output_head(model: HFModel) -> nn.Linear:
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if callable(get_output_embeddings):
        output_head = get_output_embeddings()
        if isinstance(output_head, nn.Linear):
            return output_head

    raise TypeError("Chunk Loss currently requires get_output_embeddings() to return torch.nn.Linear.")


def _install_output_head_interceptor(output_head: nn.Linear) -> None:
    if getattr(output_head, "_llamafactory_chunk_loss_enabled", False):
        return

    output_head._llamafactory_chunk_loss_original_forward = output_head.forward

    def intercepted_forward(self, hidden_states: Tensor, *args, **kwargs):
        call_state = _ACTIVE_OUTPUT_HEAD_CALL.get()
        if call_state is None:
            return self._llamafactory_chunk_loss_original_forward(hidden_states, *args, **kwargs)

        if call_state.output_head_call_count > 0:
            raise RuntimeError(f"{call_state.name} does not support more than one output head call per model forward.")
        if args or kwargs:
            raise TypeError(f"{call_state.name} does not support extra lm_head forward arguments.")

        output = call_state.compute_output(self, hidden_states)
        call_state.output_head_call_count += 1
        return output

    output_head.forward = types.MethodType(intercepted_forward, output_head)
    output_head._llamafactory_chunk_loss_enabled = True


class ChunkCrossEntropyHandler:
    """Run memory-efficient SFT loss by intercepting a causal LM's linear output head."""

    def __init__(
        self,
        model: HFModel,
        chunk_size: int | None = None,
        *,
        token_budget: int | None = None,
    ) -> None:
        self.chunk_size_policy = ChunkSizePolicy(fixed_chunk_size=chunk_size, token_budget=token_budget)
        _install_output_head_interceptor(_get_linear_output_head(model))

    def __call__(
        self,
        model: HFModel,
        model_inputs: dict[str, Tensor],
        target_labels: Tensor,
        target_weights: Tensor,
        denominator: Tensor,
    ) -> Tensor:
        def compute_output(output_head: nn.Linear, hidden_states: Tensor) -> Tensor:
            if hidden_states.ndim != 3 or hidden_states.size(0) != target_labels.size(0):
                raise ValueError(
                    "The lm_head hidden-state shape does not match the SFT targets. "
                    "This model may slice the sequence before its output head and is not supported by SFT Chunk Loss."
                )

            if hidden_states.size(1) == target_labels.size(1) + 1:
                hidden_states = hidden_states[..., :-1, :]
            elif hidden_states.size(1) != target_labels.size(1):
                raise ValueError(
                    "The lm_head hidden-state sequence length must equal the SFT target length or exceed it by one, "
                    f"but found {hidden_states.size(1)} and {target_labels.size(1)}."
                )

            chunk_size = self.chunk_size_policy.resolve(
                batch_size=hidden_states.size(0),
                sequence_length=hidden_states.size(1),
            )
            head_weight, head_bias = _get_local_output_head_parameters(output_head)
            return chunked_linear_cross_entropy(
                hidden_states=hidden_states,
                head_weight=head_weight,
                head_bias=head_bias,
                labels=target_labels,
                loss_weights=target_weights,
                chunk_size=chunk_size,
                denominator=denominator,
            )

        call_state = _OutputHeadCallState(name="SFT Chunk Loss", compute_output=compute_output)
        with _activate_output_head_call(call_state):
            outputs = model(**model_inputs)

        if call_state.output_head_call_count == 0:
            raise RuntimeError("SFT Chunk Loss did not observe a call to the patched output head.")

        loss = outputs.logits
        if loss.ndim != 0:
            raise RuntimeError(
                "SFT Chunk Loss expected the model to return the scalar output-head loss as outputs.logits, "
                f"but found shape {tuple(loss.shape)}. This model may consume or reshape logits after the output head "
                "and is not supported by the standard Chunk Loss implementation."
            )

        return loss


class ChunkLogProbHandler:
    """Return causal per-token log-probabilities and detached mean logits without full-sequence logits."""

    def __init__(
        self,
        model: HFModel,
        chunk_size: int | None = None,
        *,
        token_budget: int | None = None,
    ) -> None:
        self.chunk_size_policy = ChunkSizePolicy(fixed_chunk_size=chunk_size, token_budget=token_budget)
        _install_output_head_interceptor(_get_linear_output_head(model))

    def __call__(
        self,
        model: HFModel,
        model_inputs: dict[str, Tensor],
        target_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        per_token_logits_mean = None

        def compute_output(output_head: nn.Linear, hidden_states: Tensor) -> Tensor:
            nonlocal per_token_logits_mean
            expected_hidden_shape = (target_labels.size(0), target_labels.size(1) + 1)
            if hidden_states.ndim != 3 or hidden_states.shape[:2] != expected_hidden_shape:
                raise ValueError(
                    "The lm_head hidden-state shape does not match the causal targets. "
                    "This model may slice the sequence before its output head and is not supported by the "
                    "Chunk Log-Prob Handler."
                )

            hidden_states = hidden_states[..., :-1, :]
            chunk_size = self.chunk_size_policy.resolve(
                batch_size=hidden_states.size(0),
                sequence_length=hidden_states.size(1),
            )
            head_weight, head_bias = _get_local_output_head_parameters(output_head)
            per_token_logps, per_token_logits_mean = chunked_linear_log_probs(
                hidden_states=hidden_states,
                head_weight=head_weight,
                head_bias=head_bias,
                labels=target_labels,
                chunk_size=chunk_size,
            )
            return per_token_logps

        call_state = _OutputHeadCallState(name="Chunk Log-Prob Handler", compute_output=compute_output)
        with _activate_output_head_call(call_state):
            outputs = model(**model_inputs, use_cache=False, return_dict=True)

        if call_state.output_head_call_count == 0:
            raise RuntimeError("Chunk Log-Prob Handler did not observe a call to the patched output head.")

        per_token_logps = outputs.logits
        if per_token_logps.shape != target_labels.shape:
            raise RuntimeError(
                "Chunk Log-Prob Handler expected the model to return per-token log-probabilities as outputs.logits, "
                f"but found shape {tuple(per_token_logps.shape)}. This model may consume or reshape the output-head "
                "result and is not supported by the standard Chunk Loss implementation."
            )

        assert per_token_logits_mean is not None
        return per_token_logps, per_token_logits_mean
