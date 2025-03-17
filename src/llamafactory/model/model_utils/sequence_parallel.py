# modified from
# 1. https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/adapters/hf_adapter.py
# 2. https://github.com/jzhang38/EasyContext/
from functools import partial

import torch.distributed as dist
import transformers
import transformers.modeling_flash_attention_utils
from ring_flash_attn import zigzag_ring_flash_attn_func


def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    q_len,
    dropout=0,
    sliding_window=None,
    is_causal=True,
    group=None,
    **kwargs,
):
    attn_output = zigzag_ring_flash_attn_func(
        query_states, key_states, value_states, dropout, causal=is_causal, group=group
    )

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


def apply_sequence_parallel(model_args):
    if model_args.sequence_parallel_size == 1:
        return None  # no sequence parallelism

    # init sequence-parallel groups here
    group_this = init_sp_group(model_args.sequence_parallel_size)

    try:
        # old_flash_attention_forward = transformers.modeling_flash_attention_utils._flash_attention_forward
        if model_args.sequence_parallel_mode == "zigzag-ring":
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this)
            # assert check_params(old_flash_attention_forward, new_flash_attention_forward)
        else:
            raise NotImplementedError("Other sequence parallel modes are to be implemented.")

        # monkey patching
        transformers.modeling_flash_attention_utils._flash_attention_forward = new_flash_attention_forward
    except Exception:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
            "please pip install transformers within the versions that llama-factory requires. "
            "If the code failed with the latest version, "
            "please file an issue to https://github.com/Qihoo360/360-llama-factory"
        )

    return group_this
