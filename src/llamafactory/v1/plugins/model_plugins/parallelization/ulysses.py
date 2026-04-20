# Copyright 2025 Bytedance Ltd. and/or its affiliates. and the LlamaFactory team.
#
# This code is inspired by the Bytedance's verl library.
# https://github.com/verl-project/verl/blob/77476af84cc074edf5a6437f8d5ea418d7a54916/verl/utils/ulysses.py
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

from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from .seq_comm import SeqAllToAll4D


_ULYSSES_SEQUENCE_PARALLEL_GROUP = None


def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup):
    """Set ulysses sequence parallel process group."""
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get ulysses sequence parallel process group."""
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    """Get ulysses sequence parallel world size."""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:
    """Get ulysses sequence parallel rank."""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0


class UlyssesAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        attn_type (AttnType): attention type enum
    """

    def __init__(
        self,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        attn_fn: Optional[callable] = None,
    ) -> None:

        super().__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.attn_fn = attn_fn

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        dropout_p=0.0,
        softmax_scale=None,
        position_ids: Optional[torch.Tensor] = None,
        causal=True,
        deterministic=False,
        target_dtype=None,
        *args: Any,
    ) -> Tensor:
        """Forward.

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            attention_mask (Tensor): attention mask for the layer
            query_length (int): the length of the query sequence
            dropout_p (float, optional): dropout probability. Defaults to 0.0.
            softmax_scale (float, optional): scale factor for softmax. Defaults to None,
            position_ids (torch.Tensor, optional): position ids for the attention. Defaults to None.
            causal (bool, optional): whether to apply causal mask. Defaults to True.
            deterministic (bool, optional): whether to apply dropout in deterministic way. Defaults to False.
            target_dtype (torch.dtype, optional): target dtype for attention output. Defaults to None.
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # scatter 2, gather 1
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        if attention_mask is None:
            if position_ids is not None:
                attention_mask = torch.ones_like(position_ids).to(torch.int64)
            else:
                attention_mask = torch.ones(q.shape[0], q.shape[1], dtype=torch.int64, device=q.device)
        else:
            attention_mask = attention_mask.to(torch.int64)

        global_attention_mask = [
            torch.empty_like(attention_mask) for _ in range(get_ulysses_sequence_parallel_world_size(self.spg))
        ]
        dist.all_gather(global_attention_mask, attention_mask, group=self.spg)
        attention_mask = torch.cat(global_attention_mask, dim=1)

        context_layer = self.attn_fn(
            q,
            k,
            v,
            attention_mask,
            query_length=query_length,
            is_causal=causal,
            dropout=dropout_p,
            position_ids=position_ids,
            softmax_scale=softmax_scale,
            deterministic=deterministic,
            target_dtype=target_dtype,
        )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)

        # out e.g., [s/p::h]
        return output
