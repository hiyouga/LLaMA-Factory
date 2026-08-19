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

import sys
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

from ....accelerator.interface import Dim, DistributedInterface
from ....utils import logging
from ....utils.constants import IGNORE_INDEX
from ....utils.plugin import BasePlugin
from ....utils.types import ModelOutput
from .ulysses import (
    UlyssesAttention,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    set_ulysses_sequence_parallel_group,
)


logger = logging.get_logger(__name__)


class SequenceParallelModelPlugin(BasePlugin):
    def __call__(self, model, cp_size: int):
        return super().__call__(model, cp_size)


class SFTSequenceParallelLossPlugin(BasePlugin):
    def __call__(self, model, batch, local_loss_fn=None):
        return super().__call__(model, batch, local_loss_fn=local_loss_fn)


def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    sequence_parallel_size=1,
    dropout=0,
    deterministic=False,
    is_causal=True,
    group=None,
    mode="ulysses",
    attn_fn=None,
    target_dtype=None,
    **kwargs,
):
    if mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_fn=attn_fn)
        attn_output = dist_attn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length=query_states.shape[1] * sequence_parallel_size,
            deterministic=deterministic,
            dropout_p=dropout,
            causal=is_causal,
            position_ids=kwargs.get("position_ids", None),
            target_dtype=target_dtype,
        )
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


@SequenceParallelModelPlugin("ulysses").register()
def apply_sequence_parallel(model, cp_size: int):
    # Replace _flash_attention_forward with new_flash_attn_forward
    module = sys.modules[model.__module__]

    set_ulysses_sequence_parallel_group(DistributedInterface().get_group(Dim.CP))

    try:
        num_attention_heads, num_key_value_heads = (
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
        )
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    assert num_attention_heads % cp_size == 0, "num_attention_heads must be divisible by cp_size"
    assert num_key_value_heads % cp_size == 0 or cp_size % num_key_value_heads == 0, (
        "num_key_value_heads must be divisible by cp_size"
    )

    origin_attn = transformers.modeling_flash_attention_utils._flash_attention_forward
    new_flash_attention_forward = partial(
        new_flash_attn_forward,
        group=get_ulysses_sequence_parallel_group(),
        mode="ulysses",
        attn_fn=origin_attn,
        sequence_parallel_size=cp_size,
    )

    for module_name, module in list(sys.modules.items()):
        try:
            if (
                hasattr(module, "__file__")
                and "transformers" in module.__file__
                and getattr(module._flash_attention_forward, "__name__", "") == "_flash_attention_forward"
            ):
                module._flash_attention_forward = new_flash_attention_forward
                logger.info_rank0(
                    f"Replaced _flash_attention_forward in module {module_name} with new_flash_attn_forward for sequence parallel."
                )
        except (AttributeError, TypeError):
            continue


def _pad_and_split_model_inputs(
    data: dict[str, torch.Tensor],
    sequence_length: int,
    padded_length: int,
    rank: int,
    world_size: int,
) -> dict[str, torch.Tensor]:
    for key, value in data.items():
        if value.ndim > 1 and value.shape[-1] == sequence_length:
            value = F.pad(value, (0, padded_length - sequence_length), value=0)
            data[key] = torch.chunk(value, chunks=world_size, dim=-1)[rank].contiguous()
    return data


def prepare_sequence_parallel_sft_batch(
    batch,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare local model inputs and aligned next-token targets for one CP rank."""
    dist_interface = DistributedInterface()
    device_batch = {
        key: value.to(dist_interface.current_device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    cp_group = get_ulysses_sequence_parallel_group()
    cp_world_size = get_ulysses_sequence_parallel_world_size(cp_group)
    cp_rank = get_ulysses_sequence_parallel_rank(cp_group)
    if cp_group is None or cp_world_size <= 1:
        raise RuntimeError("Sequence-parallel batch preparation requires an initialized CP process group.")

    input_ids = device_batch["input_ids"]
    sequence_length = input_ids.shape[-1]
    max_sequence_length = torch.tensor(sequence_length, device=input_ids.device, dtype=torch.int64)
    dist.all_reduce(max_sequence_length, op=dist.ReduceOp.MAX, group=cp_group)
    max_sequence_length = int(max_sequence_length.item())
    padded_length = max_sequence_length + (cp_world_size - max_sequence_length % cp_world_size) % cp_world_size
    local_length = padded_length // cp_world_size

    model_inputs = _pad_and_split_model_inputs(
        {key: value for key, value in device_batch.items() if key not in ("labels", "loss_weights")},
        sequence_length=sequence_length,
        padded_length=padded_length,
        rank=cp_rank,
        world_size=cp_world_size,
    )
    target_labels = device_batch["labels"][..., 1:]
    target_weights = device_batch["loss_weights"][..., 1:]
    supervision_padding = padded_length - target_labels.size(-1)
    target_labels = F.pad(target_labels, (0, supervision_padding), value=IGNORE_INDEX)
    target_weights = F.pad(target_weights, (0, supervision_padding), value=0.0)
    denominator = target_weights.float().sum() + 1e-6
    local_start = cp_rank * local_length
    local_end = local_start + local_length
    return (
        model_inputs,
        target_labels[..., local_start:local_end].contiguous(),
        target_weights[..., local_start:local_end].contiguous(),
        denominator,
    )


def finalize_sequence_parallel_loss(local_loss: torch.Tensor) -> torch.Tensor:
    """Return the global value while preserving correctly scaled local-shard gradients."""
    cp_group = get_ulysses_sequence_parallel_group()
    cp_world_size = get_ulysses_sequence_parallel_world_size(cp_group)
    global_loss = local_loss.detach().clone()
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM, group=cp_group)
    return global_loss + cp_world_size * (local_loss - local_loss.detach())


def sequence_parallel_loss(model, model_inputs, target_labels, target_weights, denominator):
    """Compute the eager loss for one prepared CP shard."""
    outputs: ModelOutput = model(**model_inputs)
    logits = outputs.logits.float()
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_labels.reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    return (token_loss * target_weights.reshape(-1)).sum() / denominator


@SFTSequenceParallelLossPlugin("sequence_parallel_loss").register()
def run_sequence_parallel_sft_loss(model, batch, local_loss_fn=None):
    """Prepare one CP shard, compute its local loss, and finalize the global loss."""
    model_inputs, target_labels, target_weights, denominator = prepare_sequence_parallel_sft_batch(batch)
    if local_loss_fn is None:
        local_loss_fn = sequence_parallel_loss

    local_loss = local_loss_fn(model, model_inputs, target_labels, target_weights, denominator)
    return finalize_sequence_parallel_loss(local_loss)
