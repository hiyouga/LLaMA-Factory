# Sequence parallel utilities for LLaMA-Factory
# Legacy modes removed - use DeepSpeed ALST for modern sequence parallelism

import torch.distributed as dist

from ...extras import logging


logger = logging.get_logger(__name__)



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

    # DEPRECATED: Legacy sequence parallel modes are no longer supported
    legacy_modes = ["zigzag-ring", "ulysses", "llama3"]
    if model_args.sequence_parallel_mode in legacy_modes:
        logger.error(
            f"DEPRECATED: sequence_parallel_mode='{model_args.sequence_parallel_mode}' has been removed. "
            f"Legacy modes used monkey patching which caused version compatibility issues. "
            f"Please migrate to 'deepspeed-alst' for better performance, stability, and memory efficiency. "
            f"Set sequence_parallel_mode='deepspeed-alst' in your configuration."
        )
        raise ValueError(f"Legacy sequence parallel mode '{model_args.sequence_parallel_mode}' is no longer supported. Use 'deepspeed-alst' instead.")

    # Handle different sequence parallel modes
    if model_args.sequence_parallel_mode == "deepspeed-alst":
        # DeepSpeed ALST mode - handled by DeepSpeed, no need for monkey patching
        logger.info_rank0(f"Using DeepSpeed ALST sequence parallel mode with {model_args.sequence_parallel_size} GPUs")

        # init sequence-parallel groups here
        group_this = init_sp_group(model_args.sequence_parallel_size)
        return group_this
    else:
        raise NotImplementedError(f"Sequence parallel mode '{model_args.sequence_parallel_mode}' is not implemented.")
