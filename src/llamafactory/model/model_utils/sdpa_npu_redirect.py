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

import logging
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.utils import is_torch_npu_available


logger = logging.getLogger(__name__)

_ORIG_SDPA = F.scaled_dot_product_attention


def _to_bool_4d_mask(
    attn_mask: Optional[torch.Tensor], q_len: int, kv_len: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Normalize additive/other Hugging Face masks into a boolean mask of shape [B, 1, Q, K] (True = masked)."""
    if attn_mask is None:
        return None
    if attn_mask.dtype != torch.bool:
        attn_mask = attn_mask < 0  # additive -inf -> True
    if attn_mask.dim() == 4:
        return attn_mask[..., :q_len, :kv_len].contiguous()
    if attn_mask.dim() == 3:
        return attn_mask[:, None, :q_len, :kv_len].contiguous()
    if attn_mask.dim() == 2:
        return attn_mask[:, None, None, :kv_len].expand(-1, 1, q_len, -1).contiguous()
    return attn_mask.to(device)


def _merge_causal_mask(
    attn_mask: Optional[torch.Tensor], is_causal: bool, L: int, S: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Merge `is_causal` into the boolean/additive attention mask (True = masked)."""
    if not is_causal or L != S:
        return attn_mask
    causal_bool = torch.ones((1, 1, L, L), dtype=torch.bool, device=device).triu(1)
    if attn_mask is None:
        return causal_bool
    if attn_mask.dtype != torch.bool:
        attn_mask = attn_mask < 0
    if attn_mask.dim() == 2:
        attn_mask = attn_mask[:, None, None, :L].expand(-1, 1, L, -1).contiguous()
    elif attn_mask.dim() == 3:
        attn_mask = attn_mask[:, None, :L, :L].contiguous()
    return attn_mask | causal_bool


def _sdpa_npu_redirect(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    """A drop-in replacement for `F.scaled_dot_product_attention`.

    Automatically falls back to the native SDPA when conditions are not met.
    The NPU-fused path is only enabled when q/k/v have shape (B, N, S, D); otherwise, it falls back.
    """
    # Fall back if the feature is disabled or the conditions are not satisfied.
    if os.environ.get("NPU_FA_DISABLE", "0") == "1":
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    npu_ok = is_torch_npu_available() and (q.device.type == "npu")
    dtype_ok = q.dtype in (torch.float16, torch.bfloat16)
    shape_ok = q.dim() == 4 and k.dim() == 4 and v.dim() == 4  # 期望 BNSD
    if not (npu_ok and dtype_ok and shape_ok):
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    L, S = q.size(-2), k.size(-2)
    merged_mask = _merge_causal_mask(attn_mask, is_causal, L, S, q.device)
    mask_bool = _to_bool_4d_mask(merged_mask, q_len=L, kv_len=S, device=q.device)

    head_dim = q.size(-1)
    sc = (1.0 / math.sqrt(head_dim)) if (scale is None) else scale

    train_mode = torch.is_grad_enabled() and (dropout_p > 0)
    keep_prob = 1.0 - (dropout_p if train_mode else 0.0)

    try:
        import torch_npu

        out = torch_npu.npu_fusion_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            head_num=q.size(-3),  # N
            input_layout="BNSD",  # (B, N, S, D)
            pse=None,
            atten_mask=mask_bool,  # True = masked
            scale=sc,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            keep_prob=keep_prob,
            sync=False,
            inner_precise=0,
        )[0]
        return out
    except Exception as e:
        if os.environ.get("NPU_FA_VERBOSE", "0") == "1":
            logger.warning(f"[sdpa_npu_redirect] npu_fusion_attention failed: {e}; fallback to SDPA.")
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)


def apply_sdpa_npu_redirect(verbose: bool = True):
    """Install the redirection by pointing `F.scaled_dot_product_attention` to our implementation."""
    if getattr(F.scaled_dot_product_attention, "__wrapped_by_npu__", False):
        return
    F.scaled_dot_product_attention = _sdpa_npu_redirect
    setattr(F.scaled_dot_product_attention, "__wrapped_by_npu__", True)
    if verbose:
        logger.info("[sdpa_npu_redirect] SDPA has been redirected to Ascend npu_fusion_attention when available.")
