import math
from typing import Any, Optional

import torch
import torch.nn as nn

from ...extras import logging


logger = logging.get_logger(__name__)


def _round_scale(scale: float, step: float = 0.05, s_min: float = 1.0, s_max: Optional[float] = None) -> float:
    """Round scale to a small grid to limit cache churn."""
    if s_max is not None:
        scale = min(scale, s_max)
    scale = max(scale, s_min)
    # round to nearest multiple of `step`
    rounded = round(scale / step) * step
    # avoid floating noise like 1.6500000002
    return float(f"{rounded:.4f}")


class DynamicYarnRotaryEmbedding(nn.Module):
    """Drop-in rotary embedding that adapts YaRN scaling dynamically with input length.

    Strategy:
    - Select effective scale s = clamp(L_eff / N0, 1, s_max). L_eff = max(position_ids)+1.
    - Quantize s to a small grid (e.g., 0.05) to reduce cache variants.
    - For each s, lazily instantiate a static HF rotary embedding module (same class as the original),
      using a cloned config where rope_scaling.factor is replaced by s and max_position_embeddings=N0*s.
    - Delegate forward to the per-scale static embedding, which handles caching of cos/sin and dtype/device.

    Notes:
    - Works with architectures that expose `self_attn.rotary_emb` and expect `forward(x, position_ids)` -> (cos, sin).
    - Only active when config.rope_scaling["rope_type"] == "yarn" and config.rope_scaling["dynamic"] == True.
    """

    def __init__(
        self,
        original_rotary_module: nn.Module,
        base_config: Any,
        quant_step: float = 0.05,
    ) -> None:
        super().__init__()

        self._orig_cls = type(original_rotary_module)
        self._device = None
        for name, buf in original_rotary_module.named_buffers(recurse=False):
            if isinstance(buf, torch.Tensor):
                self._device = buf.device
                break
        # Fallback to module device if buffers missing
        if self._device is None:
            self._device = next(
                (p.device for p in original_rotary_module.parameters(recurse=False)), torch.device("cpu")
            )

        # Keep a minimal copy of scaling metadata
        rope_scaling = getattr(base_config, "rope_scaling", None) or {}
        if not isinstance(rope_scaling, dict) or rope_scaling.get("rope_type") != "yarn":
            raise ValueError("DynamicYarnRotaryEmbedding requires rope_scaling dict with rope_type='yarn'.")

        self.s_max: float = float(rope_scaling.get("factor", 1.0))
        self.N0: int = int(
            rope_scaling.get("original_max_position_embeddings") or getattr(base_config, "max_position_embeddings")
        )
        self._base_rope_kwargs: dict[str, Any] = {k: v for k, v in rope_scaling.items() if k != "dynamic"}
        self._base_config = base_config
        self._quant_step = quant_step

        # Cache of static embeddings per rounded scale
        self._embeddings_by_scale: dict[float, nn.Module] = {}

        logger.info_rank0(
            f"Enable dynamic YaRN rotary: N0={self.N0}, s_max={self.s_max}, quant_step={self._quant_step}."
        )

    def _clone_config_with_scale(self, scale: float):
        # Clone config via to_dict to avoid mutating the model-wide config
        cfg_dict = self._base_config.to_dict()
        cfg_dict["max_position_embeddings"] = int(math.ceil(self.N0 * scale))
        rope_scaling = dict(self._base_rope_kwargs)
        rope_scaling["factor"] = float(scale)
        cfg_dict["rope_scaling"] = rope_scaling

        # Reconstruct config using the same class
        cfg = self._base_config.__class__(**cfg_dict)
        return cfg

    def _get_static_embedding(self, scale: float) -> nn.Module:
        if scale in self._embeddings_by_scale:
            return self._embeddings_by_scale[scale]

        cfg = self._clone_config_with_scale(scale)
        try:
            emb = self._orig_cls(cfg, device=self._device)
        except TypeError:
            # Some classes may not accept device kwarg
            emb = self._orig_cls(cfg)
            try:
                emb.to(self._device)
            except Exception:
                pass

        self._embeddings_by_scale[scale] = emb
        logger.debug_rank0(f"Initialized static YaRN rotary for scale={scale} on device={self._device}.")
        return emb

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        seq_len: Optional[int] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Determine effective sequence length from position_ids if provided; fallback to seq_len or x.shape[1]
        if position_ids is not None:
            # position_ids may be (bsz, q_len) or 1D; take max position + 1
            L_eff = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else (seq_len or x.shape[-2])
        else:
            L_eff = seq_len or x.shape[-2]

        # Compute dynamic scale and quantize
        raw_scale = L_eff / float(self.N0)
        s = _round_scale(raw_scale, step=self._quant_step, s_min=1.0, s_max=self.s_max)

        # Delegate to per-scale static embedding
        emb = self._get_static_embedding(s)
        try:
            return emb(x, position_ids, seq_len=seq_len, **kwargs)
        except TypeError:
            return emb(x, position_ids)
