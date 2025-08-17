# modified from
# 1. https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/adapters/hf_adapter.py
# 2. https://github.com/jzhang38/EasyContext/
from functools import partial

import torch.distributed as dist
import transformers
import transformers.modeling_flash_attention_utils

from ...extras import logging


logger = logging.get_logger(__name__)


try:
    from ring_flash_attn import zigzag_ring_flash_attn_func

    RING_FLASH_ATTN_AVAILABLE = True
except ImportError:
    RING_FLASH_ATTN_AVAILABLE = False
    zigzag_ring_flash_attn_func = None

try:
    from yunchang import UlyssesAttention
    from yunchang.kernels import AttnType

    YUNCHANG_AVAILABLE = True
except ImportError:
    YUNCHANG_AVAILABLE = False
    UlyssesAttention = None
    AttnType = None


def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    q_len,
    dropout=0,
    deterministic=False,
    sliding_window=None,
    is_causal=True,
    group=None,
    mode="zigzag-ring",
    **kwargs,
):
    if mode == "zigzag-ring":
        if not RING_FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "ring-flash-attn is required for zigzag-ring mode. Please install it with: pip install ring-flash-attn flash-attn"
            )
        attn_output = zigzag_ring_flash_attn_func(
            query_states, key_states, value_states, dropout, deterministic=deterministic, causal=is_causal, group=group
        )
    elif mode == "ulysses":
        if not YUNCHANG_AVAILABLE:
            raise ImportError("yunchang is required for ulysses mode. Please install it with: pip install yunchang")
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_type=AttnType.FA)
        attn_output = dist_attn(
            query_states, key_states, value_states, deterministic=deterministic, dropout_p=dropout, causal=is_causal
        )
    elif mode == "llama3":
        # Use DeepSpeed ALST for llama3 mode - this is now implemented through ALST
        try:
            from ...model_utils.deepspeed_sequence_parallel import check_alst_requirements

            if check_alst_requirements():
                # Fallback to DeepSpeed ALST implementation
                # This should not be reached as ALST is applied at model level
                raise NotImplementedError("llama3 mode should use DeepSpeed ALST - configure sequence_parallel_mode='deepspeed-alst'")
            else:
                raise ImportError("DeepSpeed ALST requirements not met for llama3 mode. Please install DeepSpeed 0.17.4+")
        except ImportError as e:
            raise ImportError(f"llama3 mode requires DeepSpeed ALST: {e}")
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


def init_sp_group(sp_size):
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert world_size % sp_size == 0, "Total number of GPUs must be a multiple of sequence_parallel_size."

    sp_group_num = world_size // sp_size
    sp_ranks_list = [list(range(i * sp_size, i * sp_size + sp_size)) for i in range(sp_group_num)]

    sp_groups = [dist.new_group(sp_ranks_this) for sp_ranks_this in sp_ranks_list]

    global_rank_this = dist.get_rank()
    sp_idx = global_rank_this // sp_size
    return sp_groups[sp_idx]


def apply_sequence_parallel(model_args, full_determinism=False):
    if model_args.sequence_parallel_size == 1:
        return None  # no sequence parallelism

    # Check dependencies based on mode
    if model_args.sequence_parallel_mode == "zigzag-ring" and not RING_FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "ring-flash-attn is required for zigzag-ring mode. Please install it with: pip install ring-flash-attn flash-attn"
        )
    elif model_args.sequence_parallel_mode == "ulysses" and not YUNCHANG_AVAILABLE:
        raise ImportError("yunchang is required for ulysses mode. Please install it with: pip install yunchang")

    # init sequence-parallel groups here
    group_this = init_sp_group(model_args.sequence_parallel_size)

    # Handle different sequence parallel modes
    if model_args.sequence_parallel_mode == "deepspeed-alst":
        # DeepSpeed ALST mode - handled by DeepSpeed, no need for monkey patching
        logger.info_rank0(f"Using DeepSpeed ALST sequence parallel mode with {model_args.sequence_parallel_size} GPUs")
        return group_this

    try:
        # For legacy modes that require monkey patching
        if model_args.sequence_parallel_mode == "zigzag-ring":
            new_flash_attention_forward = partial(
                new_flash_attn_forward,
                group=group_this,
                mode=model_args.sequence_parallel_mode,
                deterministic=full_determinism,
            )
        elif model_args.sequence_parallel_mode == "ulysses":
            new_flash_attention_forward = partial(
                new_flash_attn_forward,
                group=group_this,
                mode=model_args.sequence_parallel_mode,
                deterministic=full_determinism,
            )
        else:
            raise NotImplementedError(f"Sequence parallel mode '{model_args.sequence_parallel_mode}' is not implemented.")

        # Apply monkey patching for legacy modes
        transformers.modeling_flash_attention_utils._flash_attention_forward = new_flash_attention_forward
        logger.info_rank0(f"Applied sequence parallel monkey patching for mode: {model_args.sequence_parallel_mode}")

    except Exception as e:
        logger.warning_rank0(
            f"Failed to apply sequence parallel monkey patching: {e}. "
            f"This may be due to transformers version {transformers.__version__} compatibility. "
            f"Consider using deepspeed-alst mode instead."
        )
        # Don't fail completely - just log the warning and continue
        pass

    return group_this
