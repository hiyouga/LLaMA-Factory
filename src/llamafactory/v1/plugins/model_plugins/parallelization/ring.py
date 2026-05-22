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

import inspect
from functools import wraps
from typing import Any, Callable, Literal, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

RingVariant = Literal["ring", "zigzag"]

_FLASH_ATTN_DTYPES = frozenset({torch.float16, torch.bfloat16})
_FLASH_ATTN_RING_PATCHED = False


def _resolve_flash_attn_dtype(query: Tensor, target_dtype: Optional[torch.dtype]) -> Optional[torch.dtype]:
    """Pick a flash-attn compatible dtype when Q/K/V are not already fp16/bf16."""
    if query.dtype in _FLASH_ATTN_DTYPES:
        return None
    if target_dtype in _FLASH_ATTN_DTYPES:
        return target_dtype
    if query.dtype != torch.float32:
        raise TypeError(
            f"Ring Flash Attention expects fp16/bf16 Q/K/V, got {query.dtype}. "
            "Enable bf16 training or cast the model to half precision."
        )

    # Full-tuning may upcast weights to fp32; transformers may pass target_dtype=float32.
    from ....utils.dtype import DtypeInterface

    if DtypeInterface.is_available("bf16"):
        return torch.bfloat16
    if DtypeInterface.is_available("fp16"):
        return torch.float16
    raise RuntimeError(
        "Ring Flash Attention requires fp16 or bf16, but neither is available on this device."
    )


def _filter_kwargs(fn: Callable, kwargs: dict) -> dict:
    allowed = inspect.signature(fn).parameters
    return {key: value for key, value in kwargs.items() if key in allowed}


def _use_ring_attn_ascend() -> bool:
    """Use Ascend NPU ring attention when running on NPU."""
    try:
        from ....accelerator.helper import is_torch_npu_available

        return is_torch_npu_available()
    except Exception:
        return hasattr(torch, "npu") and torch.npu.is_available()


def _patch_flash_attn_for_ring() -> None:
    """ring-flash-attn may pass kwargs (e.g. alibi_slopes) removed in newer flash-attn.

    Only applies to the CUDA ring-flash-attn stack; skipped on NPU (ring_attn_ascend).
    """
    global _FLASH_ATTN_RING_PATCHED
    if _FLASH_ATTN_RING_PATCHED or _use_ring_attn_ascend():
        return

    try:
        import flash_attn.flash_attn_interface as flash_attn_interface
    except ImportError:
        _FLASH_ATTN_RING_PATCHED = True
        return

    for name in ("_flash_attn_forward", "_flash_attn_backward"):
        orig_fn = getattr(flash_attn_interface, name, None)
        if orig_fn is None or getattr(orig_fn, "_llamafactory_ring_patched", False):
            continue

        @wraps(orig_fn)
        def wrapper(*args, __orig_fn=orig_fn, **kwargs):
            return __orig_fn(*args, **_filter_kwargs(__orig_fn, kwargs))

        wrapper._llamafactory_ring_patched = True
        setattr(flash_attn_interface, name, wrapper)

    _FLASH_ATTN_RING_PATCHED = True


def _get_ring_flash_attn_fn(variant: RingVariant) -> Callable:
    if _use_ring_attn_ascend():
        try:
            if variant == "zigzag":
                from ring_attn_ascend import zigzag_ring_flash_attn_func

                return zigzag_ring_flash_attn_func
            from ring_attn_ascend import ring_flash_attn_func

            return ring_flash_attn_func
        except ImportError as exc:
            raise ImportError(
                "ring-attn-ascend is required for ring context parallelism on NPU. "
                "Install with: pip install ring-attn-ascend"
            ) from exc

    try:
        _patch_flash_attn_for_ring()
        if variant == "zigzag":
            from ring_flash_attn import zigzag_ring_flash_attn_func

            return zigzag_ring_flash_attn_func
        from ring_flash_attn import ring_flash_attn_func

        return ring_flash_attn_func
    except ImportError as exc:
        raise ImportError(
            "ring-flash-attn is required for ring context parallelism on CUDA. "
            "Install with: pip install 'llamafactory[ring]'"
        ) from exc


class RingAttention(torch.nn.Module):
    """Ring attention via P2P K/V circulation and online softmax merge.

    Expects Q/K/V in layout (batch, seq_len / cp_size, num_heads, head_dim).
    """

    def __init__(
        self,
        sequence_process_group: dist.ProcessGroup = None,
        variant: RingVariant = "ring",
    ) -> None:
        super().__init__()
        self.spg = sequence_process_group
        self.variant = variant
        self.ring_fn = _get_ring_flash_attn_fn(variant)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_length: int = 0,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        position_ids: Optional[torch.Tensor] = None,
        causal: bool = True,
        deterministic: bool = False,
        target_dtype: Optional[torch.dtype] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        del attention_mask, query_length, position_ids, args, kwargs

        if self.variant == "zigzag":
            if not causal:
                raise ValueError("zigzag ring attention only supports causal=True.")
            if query.shape[1] % 2 != 0:
                raise ValueError(
                    f"local sequence length ({query.shape[1]}) must be even for zigzag ring attention. "
                    "Use cp_mode=ring or ensure padding aligns to 2 * cp_size."
                )

        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5

        flash_dtype = _resolve_flash_attn_dtype(query, target_dtype)
        if flash_dtype is not None:
            query = query.to(flash_dtype)
            key = key.to(flash_dtype)
            value = value.to(flash_dtype)

        ring_kwargs = {
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "deterministic": deterministic,
            "group": self.spg,
        }
        out = self.ring_fn(
            query,
            key,
            value,
            **_filter_kwargs(self.ring_fn, ring_kwargs),
        )
        # ring_attn_ascend (and some flash-attn builds) return (out, lse, ...).
        if isinstance(out, tuple):
            out = out[0]
        return out
