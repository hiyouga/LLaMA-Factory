from __future__ import annotations

import os
from enum import Enum

import torch

from ..extras import logging
from ..extras.packages import is_fsdp_dcp_available


logger = logging.get_logger(__name__)


class CheckpointBackend(str, Enum):
    FSDP_DCP = "fsdp_dcp"
    DEEPSPEED = "deepspeed"
    HF_FALLBACK = "hf_fallback"


def is_deepspeed_enabled() -> bool:
    try:
        from transformers.integrations import is_deepspeed_zero3_enabled

        return is_deepspeed_zero3_enabled()
    except Exception:
        return False


def is_fsdp_enabled() -> bool:
    try:
        from transformers.modeling_utils import is_fsdp_enabled as _is_fsdp_enabled

        return _is_fsdp_enabled()
    except Exception:
        return False


def select_backend() -> CheckpointBackend:
    if is_deepspeed_enabled():
        return CheckpointBackend.DEEPSPEED
    if is_fsdp_enabled() and is_fsdp_dcp_available():
        return CheckpointBackend.FSDP_DCP
    return CheckpointBackend.HF_FALLBACK


def _checkpoint_dir(output_dir: str, global_step: int) -> str:
    return os.path.join(output_dir, f"checkpoint-{global_step}")


def fsdp_dcp_save(trainer, output_dir: str) -> None:
    """Save checkpoint via PyTorch Distributed Checkpoint APIs (model-only for now)."""
    # Avoid capture during save; this should already be wrapped by caller
    ckpt_dir = output_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # Synchronize before planning
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass

    try:
        from torch.distributed import checkpoint as dist_cp
        from torch.distributed.checkpoint.filesystem import FileSystemWriter
        from torch.distributed.checkpoint.state_dict import (
            ModelStateDictOptions,
            get_state_dict,
        )

        model = trainer.model

        # Build model state dict (sharded)
        ms_opts = ModelStateDictOptions(sharded=True)
        model_sd = get_state_dict(model, options=ms_opts)

        # Compose state dict payload; extend with optimizer/scheduler later
        payload = {"model": model_sd}

        writer = FileSystemWriter(ckpt_dir)

        # Use DP group if available; else default
        process_group = None
        try:
            from accelerate.state import AcceleratorState

            state = AcceleratorState()
            process_group = getattr(state, "process_group", None)
        except Exception:
            process_group = None

        dist_cp.save(payload, storage_writer=writer, process_group=process_group)
        logger.info_rank0(f"Saved FSDP DCP checkpoint to {ckpt_dir}")

    except Exception as e:
        logger.warning_rank0(f"FSDP DCP save failed; falling back to HF: {e}")
        # Fall back to HF save
        trainer.save_model(ckpt_dir)

    # Synchronize after save
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass
