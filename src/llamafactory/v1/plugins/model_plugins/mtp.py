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

"""Multi-Token Prediction (MTP) layer support for the v1 architecture.

This module is adapted from the FSDP2 MTP implementation in MindSpeed-LLM
(https://gitcode.com/Ascend/MindSpeed-LLM, ``mindspeed_llm/fsdp2/models/common/mtp.py``)
and reworked to be HuggingFace-transformers generic so that it can be attached to any
decoder-only causal LM that follows the Llama/Qwen3 layout (``model.model.layers``,
``model.model.rotary_emb``, ``model.model.norm``, ``model.lm_head``).

Design overview
---------------
* ``MultiTokenPredictionBlock`` holds ``num_layers`` prediction heads. Each head reuses
  the base model's decoder layer class. The block owns shared ``enorm``/``hnorm`` norms,
  ``e_proj``/``h_proj`` projections and a ``final_layernorm`` (exactly as in MindSpeed).
* The block is attached to a ``ForCausalLM`` model as ``model.mtp`` by ``MTPModelPlugin``,
  which also patches ``model.forward`` so that the model output carries ``mtp_logits``
  (a list of per-head logits, one per MTP head).
* The actual loss is *not* computed inside the model. It is computed by the trainer
  (non-CP path) or by the sequence-parallel loss plugin (CP path) through the shared
  ``compute_mtp_loss`` helper. This keeps the loss weighting (``loss_weights``) and the
  context-parallel all-gather logic in a single place.

Conventions
-----------
* The main model head predicts token ``p + 1`` from hidden state at position ``p``.
* MTP head ``k`` (0-indexed) predicts token ``p + k + 2``. The prediction of head ``k``
  uses ``mtp_logits[k][:, :-(k + 2)]`` against ``labels[:, k + 2:]``.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn  # noqa: F401  required for `dist.nn.all_gather` (autograd-friendly)
import torch.nn as nn
import torch.nn.functional as F

from ...utils import logging
from ...utils.plugin import BasePlugin


if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel

    from ...utils.types import PluginConfig


logger = logging.get_logger(__name__)


def roll_tensor(
    tensor: torch.Tensor, shifts: int = -1, dim: int = -1, fill_value: float = 0.0
) -> torch.Tensor:
    """Roll a tensor along ``dim`` and fill the wrapped positions with ``fill_value``.

    This mirrors ``roll_tensor`` in MindSpeed-LLM. With ``shifts=-1, dim=-1`` it shifts
    the sequence one step to the left and sets the last position to ``fill_value``.
    """
    rolled = torch.roll(tensor, shifts=shifts, dims=dim)
    rolled.select(dim, shifts).fill_(fill_value)
    return rolled


class MultiTokenPredictionLayer(nn.Module):
    """A single MTP head: one decoder layer reused from the base model."""

    def __init__(self, config: "PretrainedConfig", layer_idx: int, layer_cls: type[nn.Module]) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer = layer_cls(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            use_cache=False,
            **kwargs,
        )
        return hidden_states


class MultiTokenPredictionBlock(nn.Module):
    """Container of ``num_layers`` MTP heads.

    Args:
        config: The base model ``PretrainedConfig``.
        layer_cls: Decoder layer class to reuse for every MTP head
            (e.g. ``LlamaDecoderLayer`` / ``Qwen3DecoderLayer``).
        norm_cls: RMS-norm class used by the base model (e.g. ``LlamaRMSNorm``).
        num_layers: Number of MTP heads (``K``).
        embed_tokens: The base model input embedding module.
        rotary_emb: The base model rotary embedding module.
        output_layer: The base model ``lm_head`` module, shared by all MTP heads.
    """

    def __init__(
        self,
        config: "PretrainedConfig",
        layer_cls: type[nn.Module],
        norm_cls: type[nn.Module],
        num_layers: int,
        embed_tokens: nn.Module,
        rotary_emb: nn.Module,
        output_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        self.output_layer = output_layer
        self.mtp_start_layer_idx = config.num_hidden_layers

        # MTP heads are brand-new decoder layers. Some models index per-layer config
        # lists (e.g. ``config.layer_types[layer_idx]``) by ``layer_idx``, which would
        # raise out of range for an index >= ``num_hidden_layers``. Use a valid in-range
        # index (the last decoder layer) so the cloned layer behaves like a normal one.
        safe_layer_idx = max(0, config.num_hidden_layers - 1)

        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + i): MultiTokenPredictionLayer(
                    config, safe_layer_idx, layer_cls
                )
                for i in range(num_layers)
            }
        )
        rms_eps = getattr(config, "rms_norm_eps", 1e-6)
        hidden_size = config.hidden_size
        self.enorm = norm_cls(hidden_size, eps=rms_eps)
        self.hnorm = norm_cls(hidden_size, eps=rms_eps)
        self.e_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.final_layernorm = norm_cls(hidden_size, eps=rms_eps)

        # Reuse the base model weight init scheme if available.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = getattr(self.config, "initializer_range", 0.02)
        for module in (self.e_proj, self.h_proj):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> list[torch.Tensor]:
        """Run all MTP heads and return per-head logits.

        Args:
            hidden_states: Last decoder layer output of the main model (pre final-norm),
                shape ``(batch, seq_len, hidden_size)``.
            input_ids: Input ids of the main model, shape ``(batch, seq_len)``.
            attention_mask: 2D attention mask from the main model, shape ``(batch, seq_len)``.
            position_ids: Position ids, shape ``(batch, seq_len)``.

        Returns:
            A list of length ``num_layers``; each item is the per-head logits of shape
            ``(batch, seq_len, vocab_size)``.
        """
        from transformers.masking_utils import create_causal_mask

        batch_size, seq_len, _ = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        # Shift input ids by one to obtain the embedding of the "next" token.
        shifted_input_ids = roll_tensor(input_ids, shifts=-1, dim=-1, fill_value=0)
        input_embeds = self.embed_tokens(shifted_input_ids)

        # Causal mask for the MTP decoder layers (same construction as the base model).
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(input_embeds, position_ids=position_ids)
        cache_position = torch.arange(seq_len, device=hidden_states.device)

        # Combine the main hidden state with the next-token embedding.
        hidden_states = self.hnorm(hidden_states) + self.e_proj(self.enorm(input_embeds))

        all_mtp_logits: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            hidden_states = self.layers[str(self.mtp_start_layer_idx + layer_idx)](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )
            hidden_states = self.final_layernorm(hidden_states)
            mtp_logits = self.output_layer(hidden_states)
            all_mtp_logits.append(mtp_logits)

        return all_mtp_logits


def compute_mtp_loss(
    mtp_logits: list[torch.Tensor],
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    ignore_index: int = -100,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute the averaged MTP loss across all heads.

    Each head ``k`` predicts token ``p + k + 2`` from position ``p``. The loss of head
    ``k`` is the ``loss_weights``-weighted cross-entropy mean over valid positions. The
    returned scalar is the plain mean over the ``K`` heads (the caller applies the
    MTP loss scaling factor, matching MindSpeed-LLM).

    When ``cp_group`` is not ``None`` (context parallelism / Ulysses), the per-head loss
    is computed on the full sequence by all-gathering labels / loss_weights / log-probs
    across the CP group, exactly like the single-head ``sequence_parallel_loss`` plugin.

    Args:
        mtp_logits: List of ``K`` tensors, each ``(batch, seq_len, vocab_size)``. The
            sequence dimension is the *local* chunk under CP.
        labels: ``(batch, seq_len)`` (local chunk under CP).
        loss_weights: ``(batch, seq_len)`` (local chunk under CP).
        ignore_index: Label index to ignore (defaults to -100).
        cp_group: Context-parallel process group, or ``None`` for the non-CP path.

    Returns:
        Scalar MTP loss (mean over heads).
    """
    num_heads = len(mtp_logits)
    if num_heads == 0:
        return torch.tensor(0.0, device=labels.device, dtype=torch.float32)

    cp_size = dist.get_world_size(cp_group) if (cp_group is not None and dist.is_initialized()) else 1

    total_loss = None
    for k, logits_k in enumerate(mtp_logits):
        shift = k + 2
        if cp_size > 1:
            head_loss = _cp_head_loss(logits_k, labels, loss_weights, shift, ignore_index, cp_group, cp_size)
        else:
            head_loss = _local_head_loss(logits_k, labels, loss_weights, shift, ignore_index)
        total_loss = head_loss if total_loss is None else total_loss + head_loss

    return total_loss / num_heads


def _local_head_loss(
    logits_k: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    shift: int,
    ignore_index: int,
) -> torch.Tensor:
    """Per-head weighted CE mean for the non-CP path."""
    if logits_k.size(1) <= shift:
        return torch.tensor(0.0, device=logits_k.device, dtype=torch.float32)

    batch_size, seq_len, vocab_size = logits_k.shape
    pred = logits_k[:, :-shift, :].float().reshape(-1, vocab_size)
    tgt = labels[:, shift:].contiguous().reshape(-1)
    weights = loss_weights[:, shift:].contiguous().reshape(-1)

    log_probs = -F.cross_entropy(pred, tgt, reduction="none", ignore_index=ignore_index).view(batch_size, -1)
    weights = weights.view(batch_size, -1)
    return (-log_probs * weights).sum() / (weights.sum() + 1e-6)


def _cp_head_loss(
    logits_k: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    shift: int,
    ignore_index: int,
    cp_group: dist.ProcessGroup,
    cp_size: int,
) -> torch.Tensor:
    """Per-head weighted CE mean for the Ulysses context-parallel path.

    Mirrors ``sequence_parallel_loss`` but for an MTP head whose target is shifted by
    ``shift`` positions. Local ``logits_k`` cover the local sequence chunk; labels and
    loss_weights are all-gathered to the full sequence, shifted, and re-chunked so that
    each local logit is aligned with its (globally shifted) target.
    """
    batch_size, local_len, vocab_size = logits_k.shape

    # All-gather labels and loss_weights across the CP group to reconstruct the full seq.
    global_labels = [torch.empty_like(labels) for _ in range(cp_size)]
    dist.all_gather(global_labels, labels, group=cp_group)
    global_labels = torch.cat(global_labels, dim=1).contiguous()

    global_loss_weights = [torch.empty_like(loss_weights) for _ in range(cp_size)]
    dist.all_gather(global_loss_weights, loss_weights, group=cp_group)
    global_loss_weights = torch.cat(global_loss_weights, dim=1).contiguous()

    cp_rank = dist.get_rank(cp_group)
    full_len = global_labels.size(1)

    # Shift labels by ``shift`` to obtain targets for head k, pad to ``full_len`` and
    # take the local chunk so that it aligns one-to-one with the local logits.
    shift_labels = global_labels[:, shift:]
    shift_labels = F.pad(shift_labels, (0, shift), value=ignore_index)
    shift_labels = torch.chunk(shift_labels, chunks=cp_size, dim=1)[cp_rank].contiguous()

    shift_logits = logits_k.float().reshape(-1, vocab_size)
    shift_labels = shift_labels.reshape(-1)
    log_probs = -F.cross_entropy(shift_logits, shift_labels, reduction="none", ignore_index=ignore_index)
    log_probs = log_probs.view(batch_size, local_len)

    # All-gather log_probs across the CP group and trim to the valid prefix.
    global_log_probs = dist.nn.all_gather(log_probs, group=cp_group)
    global_log_probs = torch.cat(global_log_probs, dim=1).contiguous()
    global_log_probs = global_log_probs[:, : full_len - shift].contiguous()

    weights = global_loss_weights[:, shift:].contiguous()
    return (-global_log_probs * weights).sum() / (weights.sum() + 1e-6)


class MTPModelPlugin(BasePlugin):
    """Plugin that grafts a ``MultiTokenPredictionBlock`` onto a causal LM."""

    def __call__(self, model: "PreTrainedModel", mtp_config: "PluginConfig") -> "PreTrainedModel":
        return apply_mtp(model, mtp_config)


def _get_inner_model(model: "PreTrainedModel") -> nn.Module:
    """Return the inner transformer model (``model.model`` for causal LMs)."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError(
            "MTP currently expects a decoder-only causal LM with a `model.model` attribute "
            "(Llama/Qwen3/Mistral-style). Got incompatible model."
        )
    return inner


def _resolve_layer_cls(model: "PreTrainedModel") -> tuple[type[nn.Module], type[nn.Module]]:
    """Resolve ``(decoder_layer_cls, norm_cls)`` from the base model.

    The decoder layer class is taken from the first entry of ``model.model.layers``.
    The norm class is taken from ``model.model.norm``. Both are standard for
    Llama/Qwen3/Mistral-style models. This avoids importing the FSDP2 plugin (and its
    extra dependencies) at module-import time.
    """
    inner = _get_inner_model(model)
    layers = getattr(inner, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Cannot find decoder layers (model.model.layers) to clone for MTP.")
    layer_cls = type(layers[0])

    norm = getattr(inner, "norm", None)
    if norm is None:
        raise ValueError("Cannot find the final norm (model.model.norm) to clone for MTP.")
    norm_cls = type(norm)
    return layer_cls, norm_cls


def apply_mtp(model: "PreTrainedModel", mtp_config: "PluginConfig") -> "PreTrainedModel":
    """Attach an MTP block to ``model`` and patch its forward to emit ``mtp_logits``."""
    num_layers = int(mtp_config.get("num_layers", 1))
    if num_layers <= 0:
        return model

    layer_cls, norm_cls = _resolve_layer_cls(model)
    inner = _get_inner_model(model)
    embed_tokens = model.get_input_embeddings()
    rotary_emb = getattr(inner, "rotary_emb", None)
    if rotary_emb is None:
        raise ValueError("MTP requires the base model to expose `model.model.rotary_emb`.")
    output_layer = getattr(model, "lm_head", None)
    if output_layer is None:
        raise ValueError("MTP requires the base model to expose `lm_head`.")

    block = MultiTokenPredictionBlock(
        config=model.config,
        layer_cls=layer_cls,
        norm_cls=norm_cls,
        num_layers=num_layers,
        embed_tokens=embed_tokens,
        rotary_emb=rotary_emb,
        output_layer=output_layer,
    )
    # Match the base model parameter dtype (e.g. bf16) so the grafted layers and the
    # shared embedding / lm_head agree on dtype.
    try:
        param_dtype = next(model.parameters()).dtype
        block = block.to(param_dtype)
    except StopIteration:
        pass
    model.mtp = block
    model.config.mtp_num_layers = num_layers
    model.config.mtp_loss_scaling_factor = float(mtp_config.get("loss_scale", 0.3))

    _patch_forward(model)

    logger.info_rank0(
        f"Enabled Multi-Token Prediction with {num_layers} head(s) "
        f"(loss_scale={model.config.mtp_loss_scaling_factor})."
    )
    return model


def _patch_forward(model: "PreTrainedModel") -> None:
    """Patch ``model.forward`` so its output carries ``mtp_logits`` during training."""
    import types

    orig_forward = model.forward

    def mtp_forward(self, *args, **kwargs):
        # We need the pre-norm last hidden state, which is the last entry of
        # ``outputs.hidden_states``. Force the inner model to return it.
        kwargs["output_hidden_states"] = True
        outputs = orig_forward(*args, **kwargs)

        # MTP logits are only needed for loss computation during training.
        mtp_block = getattr(self, "mtp", None)
        if mtp_block is not None and self.training:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is not None:
                hidden = hidden_states[-1]
            else:
                hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is not None:
                input_ids = kwargs.get("input_ids", args[0] if args else None)
                attention_mask = kwargs.get("attention_mask", None)
                position_ids = kwargs.get("position_ids", None)
                mtp_logits = mtp_block(
                    hidden,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                outputs.mtp_logits = mtp_logits
        return outputs

    model.forward = types.MethodType(mtp_forward, model)
