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

import functools
from typing import TYPE_CHECKING

import torch

from ...extras import logging
from ...extras.constants import AttentionFunction
from ...extras.packages import is_torch_version_greater_than


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


# flash-attn-4 (CuTe-DSL backend, v4.0.0.beta21) crashes in its backward-PREPROCESS kernel for any
# attention head_dim that is not a multiple of 32. That kernel pads head_dim to a multiple of 32
# (flash_attn/cute/flash_bwd_preprocess.py:68, `hdim_multiple_of = 32`); when the real head_dim differs
# from the padded value it takes an out-of-bounds copy-predicate path (`check_hdim_v_oob`, line 71) and
# passes the predicate to `cute.copy` WITHOUT slicing it per row-tile (lines 367-369, 381-382), unlike
# the forward and main-backward kernels which slice it (e.g. flash_fwd.py:478). The result is a CuTe
# shape mismatch ("'cute.copy' op expects pred to have compatible shape with (1,(3)) but got (1,(2,3))").
# The forward kernel pads to a multiple of 16 and slices its predicate, so it is unaffected -- the bug is
# backward-only. Verified on B30Z (sm_103a): head_dim 64/128/256 train fine, 72/80 abort at first backward.
# Vision towers commonly hit this (Qwen3-VL / Qwen3.5 ViT: hidden 1152 / 16 heads = head_dim 72), so we
# keep such vision towers on fa2 and route only the (safe) language model to fa4.
_FA4_HEAD_DIM_MULTIPLE = 32


def _sub_config_head_dim(sub_config: "PretrainedConfig") -> "int | None":
    r"""Return the attention head_dim of a sub-config, deriving it when not stored explicitly.

    Vision configs (e.g. Qwen*VL) expose ``num_heads`` rather than ``num_attention_heads`` and rarely
    carry a ``head_dim`` attribute, so fall back to ``hidden_size // num_heads``.
    """
    head_dim = getattr(sub_config, "head_dim", None)
    if head_dim:
        return int(head_dim)

    hidden_size = getattr(sub_config, "hidden_size", None)
    num_heads = getattr(sub_config, "num_attention_heads", None) or getattr(sub_config, "num_heads", None)
    if hidden_size and num_heads:
        return int(hidden_size) // int(num_heads)

    return None


def _fa4_vision_needs_fa2(config: "PretrainedConfig") -> bool:
    r"""Whether the model has a vision tower whose head_dim is unsupported by the fa4 backward kernel."""
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        return False

    head_dim = _sub_config_head_dim(vision_config)
    return head_dim is not None and head_dim % _FA4_HEAD_DIM_MULTIPLE != 0


def _patch_fa4_varlen_int_seqlen() -> None:
    r"""Coerce flash-attn-4 varlen ``max_seqlen_q``/``max_seqlen_k`` from tensor to Python int.

    On packed / position-ids paths transformers passes ``cu_seqlens.diff().max()`` -- a CUDA tensor -- as
    fa4's ``max_seqlen_q``/``max_seqlen_k``, which are documented as ``Optional[int]``. fa4's cute backward
    builds its JIT compile-key from ``max_seqlen``; a tensor in that key hashes by object identity rather
    than value, so every step misses the compile cache and recompiles the backward kernel (a ~12x per-step
    slowdown vs. fa2). Coercing to int makes the key stable (compile once).

    Wrapping fa4's own entry point is the true API boundary: it catches every fa4 varlen call, honors the
    documented ``int`` contract, and leaves transformers and the installed wheel untouched. It is fa4-only
    and self-skipping -- a no-op when ``flash_attn.cute`` is unavailable, and idempotent via a
    ``_llamafactory_int_seqlen_patched`` flag.
    """
    try:
        import flash_attn.cute as cute_pkg
        from flash_attn.cute import interface as cute_iface
    except Exception:  # fa4 not installed; nothing to patch.
        return

    orig = getattr(cute_iface, "flash_attn_varlen_func", None)
    if orig is None or getattr(orig, "_llamafactory_int_seqlen_patched", False):
        return

    # Public signature: flash_attn_varlen_func(q, k, v, qv=None, cu_seqlens_q=None,
    #   cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None, ...) -> positional indices 6, 7.
    max_seqlen_q_pos, max_seqlen_k_pos = 6, 7

    @functools.wraps(orig)
    def flash_attn_varlen_func_int_seqlen(*args, **kwargs):
        for name in ("max_seqlen_q", "max_seqlen_k"):
            value = kwargs.get(name)
            if isinstance(value, torch.Tensor):
                kwargs[name] = int(value.item())

        if args:
            args = list(args)
            for pos in (max_seqlen_q_pos, max_seqlen_k_pos):
                if len(args) > pos and isinstance(args[pos], torch.Tensor):
                    args[pos] = int(args[pos].item())

            args = tuple(args)

        return orig(*args, **kwargs)

    flash_attn_varlen_func_int_seqlen._llamafactory_int_seqlen_patched = True
    # Rebind on the interface module and the package re-export (transformers lazy-imports the object via
    # `from flash_attn.cute import flash_attn_varlen_func`).
    cute_iface.flash_attn_varlen_func = flash_attn_varlen_func_int_seqlen
    cute_pkg.flash_attn_varlen_func = flash_attn_varlen_func_int_seqlen
    logger.info_rank0("Patched FlashAttention-4 varlen max_seqlen to int (avoids per-step backward recompile).")


def configure_attn_implementation(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    from transformers.utils import is_flash_attn_2_available

    if getattr(config, "model_type", None) == "gpt_oss":
        from transformers.integrations.hub_kernels import load_and_register_kernel

        flash_attn3_kernel = "kernels-community/vllm-flash-attn3"
        load_and_register_kernel(flash_attn3_kernel)
        setattr(config, "_attn_implementation", flash_attn3_kernel)
        setattr(config, "_attn_implementation_internal", flash_attn3_kernel)
        model_args.flash_attn = AttentionFunction.FA3

        logger.info_rank0("Using FlashAttention-3 with attention sink for the gpt-oss model.")
        return

    if getattr(config, "model_type", None) == "gemma2":
        if model_args.flash_attn == AttentionFunction.AUTO or model_args.flash_attn == AttentionFunction.FA2:
            if is_flash_attn_2_available():
                if model_args.flash_attn != AttentionFunction.FA2:
                    logger.warning_rank0("Gemma 2 should use flash attention 2, change `flash_attn` to fa2.")
                    model_args.flash_attn = AttentionFunction.FA2
            else:
                logger.warning_rank0("FlashAttention-2 is not installed, use eager attention.")
                model_args.flash_attn = AttentionFunction.DISABLED
        elif model_args.flash_attn == AttentionFunction.SDPA:
            logger.warning_rank0(
                "Gemma-2 should use soft-capping attention, while the SDPA attention does not support it."
            )

    if getattr(config, "model_type", None) in ["youtu", "youtu_vl"]:
        if model_args.flash_attn in (AttentionFunction.AUTO, AttentionFunction.SDPA):
            logger.warning_rank0("Youtu-VL does not support SDPA, forcing eager attention.")
            model_args.flash_attn = AttentionFunction.DISABLED

    if model_args.flash_attn == AttentionFunction.AUTO:
        return

    elif model_args.flash_attn == AttentionFunction.DISABLED:
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == AttentionFunction.SDPA:
        if not is_torch_version_greater_than("2.1.1"):
            logger.warning_rank0("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == AttentionFunction.FA2:
        from transformers import is_torch_npu_available

        if not (is_flash_attn_2_available() or is_torch_npu_available()):
            logger.warning_rank0("FlashAttention-2 is not installed.")
            return

        requested_attn_implementation = "flash_attention_2"
    elif model_args.flash_attn == AttentionFunction.FA4:
        try:
            from transformers.utils import is_flash_attn_4_available
        except ImportError:
            logger.warning_rank0("This transformers version does not support FlashAttention-4; please upgrade.")
            return

        if not is_flash_attn_4_available():
            logger.warning_rank0("FlashAttention-4 is not installed or unavailable.")
            return

        # fa4's cute varlen entry point is typed max_seqlen_q/k: Optional[int], but transformers passes a
        # CUDA tensor on the packed/position_ids path; a tensor in fa4's JIT compile-key hashes by identity
        # and recompiles the backward kernel every step (~12x slowdown). Coerce it to int.
        _patch_fa4_varlen_int_seqlen()
        requested_attn_implementation = "flash_attention_4"
    else:
        raise NotImplementedError(f"Unknown attention type: {model_args.flash_attn}")

    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "kimi_vl":
        setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "youtu_vl":
        setattr(config, "attn_implementation", requested_attn_implementation)
        setattr(config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "vision_config"):
            setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    elif requested_attn_implementation == "flash_attention_4" and _fa4_vision_needs_fa2(config):
        # Route the vision tower to fa2 (unsupported head_dim on the fa4 backward kernel) and the language
        # model to fa4. transformers resolves this dict form onto the sub-configs via `config.sub_configs`.
        setattr(
            config,
            "_attn_implementation",
            {"": "flash_attention_4", "text_config": "flash_attention_4", "vision_config": "flash_attention_2"},
        )
        logger.warning_rank0(
            f"Vision tower head_dim={_sub_config_head_dim(config.vision_config)} is unsupported by the "
            "FlashAttention-4 backward kernel; routing the vision tower to FlashAttention-2 and the language "
            "model to FlashAttention-4."
        )
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info_rank0("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "flash_attention_4":
        logger.info_rank0("Using FlashAttention-4 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info_rank0("Using torch SDPA for faster training and inference.")
    else:
        logger.info_rank0("Using vanilla attention implementation.")
