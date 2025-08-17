# -*- coding: utf-8 -*-
"""
全局把 F.scaled_dot_product_attention 重定向为 Ascend NPU 的 torch_npu.npu_fusion_attention。
- 不修改任何 Module/forward，不触碰 _parameters，天然兼容 ZeRO-3/offload。
- 仅在 NPU + 半精 (fp16/bf16) 且未禁用开关时启用；否则回退原生 SDPA。
- 自动把 is_causal 并入布尔 mask（NPU算子用 True=mask）。
"""

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# 备份原始 SDPA
_ORIG_SDPA = F.scaled_dot_product_attention

def _npu_available() -> bool:
    # 尽量稳健地判断 NPU 可用
    try:
        import torch_npu  # noqa: F401
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False

def _to_bool_4d_mask(attn_mask: Optional[torch.Tensor],
                     q_len: int,
                     kv_len: int,
                     device: torch.device) -> Optional[torch.Tensor]:
    """把 HF 的加性/其他 mask 统一成 [B,1,Q,K] 的 bool 掩码（True=屏蔽）"""
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
    # 其他形状：尽力直接返回
    return attn_mask.to(device)

def _merge_causal_mask(attn_mask: Optional[torch.Tensor],
                       is_causal: bool,
                       L: int,
                       S: int,
                       device: torch.device) -> Optional[torch.Tensor]:
    """把 is_causal 并到布尔/加性 mask 中（True=mask）。"""
    if not is_causal or L != S:
        return attn_mask
    # 生成上三角（不含对角），True=mask
    causal_bool = torch.ones((1, 1, L, L), dtype=torch.bool, device=device).triu(1)
    if attn_mask is None:
        return causal_bool
    # 如果 attn_mask 是加性，会在 _to_bool_4d_mask 内转 bool；这里做逻辑或
    if attn_mask.dtype != torch.bool:
        attn_mask = attn_mask < 0
    # 广播到 4D 再 or
    if attn_mask.dim() == 2:
        attn_mask = attn_mask[:, None, None, :L].expand(-1, 1, L, -1).contiguous()
    elif attn_mask.dim() == 3:
        attn_mask = attn_mask[:, None, :L, :L].contiguous()
    return (attn_mask | causal_bool)

def _sdpa_npu_redirect(q: torch.Tensor,
                       k: torch.Tensor,
                       v: torch.Tensor,
                       attn_mask: Optional[torch.Tensor] = None,
                       dropout_p: float = 0.0,
                       is_causal: bool = False,
                       scale: Optional[float] = None):
    """
    作为 F.scaled_dot_product_attention 的替代实现。
    条件不满足时自动回退原生 SDPA。
    仅支持 q/k/v 形状为 (B, N, S, D) 时走 NPU 融合，否则回退。
    """
    # 关闭开关或条件不满足 -> 回退
    if os.environ.get("NPU_FA_DISABLE", "0") == "1":
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale)

    npu_ok = _npu_available() and (q.device.type == "npu")
    dtype_ok = q.dtype in (torch.float16, torch.bfloat16)
    shape_ok = (q.dim() == 4 and k.dim() == 4 and v.dim() == 4)  # 期望 BNSD
    if not (npu_ok and dtype_ok and shape_ok):
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale)

    # 把 is_causal 并入 mask
    L, S = q.size(-2), k.size(-2)
    merged_mask = _merge_causal_mask(attn_mask, is_causal, L, S, q.device)
    # 转为 NPU 期待的布尔 4D mask
    mask_bool = _to_bool_4d_mask(merged_mask, q_len=L, kv_len=S, device=q.device)

    # 计算缩放、keep_prob
    head_dim = q.size(-1)
    sc = (1.0 / math.sqrt(head_dim)) if (scale is None) else scale
    # 训练时才保留 dropout；注意：我们没有 self.training，只能按 grad 是否开启近似判断
    train_mode = torch.is_grad_enabled() and (dropout_p > 0)
    keep_prob = 1.0 - (dropout_p if train_mode else 0.0)

    try:
        import torch_npu
        out = torch_npu.npu_fusion_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            head_num=q.size(-3),             # N
            input_layout="BNSD",             # (B, N, S, D)
            pse=None,
            atten_mask=mask_bool,            # True = masked
            scale=sc,
            pre_tockens=2147483647,          # 全可见，若需滑窗可调
            next_tockens=2147483647,
            keep_prob=keep_prob,
            sync=False,
            inner_precise=0,
        )[0]
        return out
    except Exception as e:
        # 任意异常都安全回退
        if os.environ.get("NPU_FA_VERBOSE", "0") == "1":
            logger.warning(f"[sdpa_npu_redirect] npu_fusion_attention failed: {e}; fallback to SDPA.")
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale)

def apply_sdpa_npu_redirect(verbose: bool = True):
    """一次性装载：把 F.scaled_dot_product_attention 指向我们的重定向实现。"""
    # 避免重复装载
    if getattr(F.scaled_dot_product_attention, "__wrapped_by_npu__", False):
        return
    F.scaled_dot_product_attention = _sdpa_npu_redirect
    setattr(F.scaled_dot_product_attention, "__wrapped_by_npu__", True)
    if verbose:
        logger.info("[sdpa_npu_redirect] SDPA has been redirected to Ascend npu_fusion_attention when available.")
