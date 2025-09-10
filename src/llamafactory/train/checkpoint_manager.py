from __future__ import annotations

import os
from enum import Enum
from typing import Any

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


def _get_dp_process_group():
    try:
        from accelerate.state import AcceleratorState

        state = AcceleratorState()
        return getattr(state, "process_group", None)
    except Exception:
        return None


def _is_nonstandard_optimizer(optimizer: torch.optim.Optimizer | None) -> bool:
    if optimizer is None:
        return False
    try:
        mod = optimizer.__class__.__module__.lower()
        name = optimizer.__class__.__name__.lower()
    except Exception:
        return False
    if "bitsandbytes" in mod or "bnb" in mod:
        return True
    if "deepspeed" in mod:
        return True
    if "paged" in name or "8bit" in name or "8_bit" in name:
        return True
    # Some paged optimizer wrappers expose an attribute
    if getattr(optimizer, "is_paged", False):
        return True
    return False


def fsdp_dcp_save(trainer, output_dir: str) -> None:
    """Save checkpoint via PyTorch Distributed Checkpoint APIs.

    - Saves model via sharded DCP state dict.
    - Attempts to save optimizer via torch.distributed.checkpoint.optim.get_optimizer_state_dict when available,
      otherwise falls back to optimizer.state_dict(). Skips non-standard optimizers (paged/8-bit) with a warning
      unless args.save_only_model is True, in which case optimizer is skipped silently.
    - Saves scheduler via .state_dict() if present and save_only_model is False.
    - Uses the data-parallel process group when available; syncs before and after.
    """
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

        # Version‑compatible DCP imports
        # - Torch 2.5+ often has ModelStateDictOptions; older/newer use StateDictOptions+ShardedStateDictConfig
        try:  # Prefer new model helpers
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict as dcp_get_model_state_dict,
            )
        except Exception:  # Extremely old versions (unlikely in our matrix)
            from torch.distributed.checkpoint.state_dict import (
                get_state_dict as dcp_get_model_state_dict,  # type: ignore
            )

        # Options shims
        try:
            from torch.distributed.checkpoint.state_dict import (
                ModelStateDictOptions as DCPStateDictOptions,  # type: ignore
            )

            def _make_model_options():
                return DCPStateDictOptions(sharded=True)  # type: ignore[arg-type]

        except Exception:
            from torch.distributed.checkpoint.state_dict import (  # type: ignore
                ShardedStateDictConfig,
                StateDictOptions,
            )

            def _make_model_options():
                return StateDictOptions(state_dict_config=ShardedStateDictConfig())

        # Optimizer helper: may be absent in some torch versions
        try:
            from torch.distributed.checkpoint.state_dict import (
                get_optimizer_state_dict as dcp_get_optim_sd,  # type: ignore
            )
        except Exception:
            dcp_get_optim_sd = None  # type: ignore

        model = trainer.model
        optimizer = getattr(trainer, "optimizer", None)
        scheduler = getattr(trainer, "lr_scheduler", None)
        save_only_model = getattr(trainer.args, "save_only_model", False)

        # Build model state dict (sharded)
        ms_opts = _make_model_options()
        model_sd = dcp_get_model_state_dict(model, options=ms_opts)

        payload: dict[str, Any] = {"model": model_sd}

        # Optimizer state (optional)
        include_opt = (not save_only_model) and optimizer is not None
        if include_opt:
            if _is_nonstandard_optimizer(optimizer):
                logger.warning_rank0(
                    "Detected non-standard optimizer (paged/8-bit/bitsandbytes). Skipping optimizer state save."
                )
            else:
                opt_sd: Any
                try:
                    if dcp_get_optim_sd is not None:
                        opt_sd = dcp_get_optim_sd(optimizer)
                        logger.info_rank0("Saving optimizer state via DCP optimizer API.")
                    else:
                        raise RuntimeError("DCP optimizer helpers unavailable")
                except Exception:
                    # Fallback: regular state dict
                    opt_sd = optimizer.state_dict()
                    logger.info_rank0("Saving optimizer state via standard state_dict() fallback.")
                payload["optimizer"] = opt_sd

        # Scheduler state (optional, lightweight)
        include_sched = (not save_only_model) and (scheduler is not None)
        if include_sched:
            try:
                payload["scheduler"] = scheduler.state_dict()
            except Exception as e:
                logger.warning_rank0(f"Failed to get scheduler state_dict; skipping scheduler save: {e}")

        writer = FileSystemWriter(ckpt_dir)

        # Use DP group if available; else default
        process_group = _get_dp_process_group()

        dist_cp.save(payload, storage_writer=writer, process_group=process_group)
        logger.info_rank0(f"Saved FSDP DCP checkpoint to {ckpt_dir}")

    except Exception as e:
        logger.warning_rank0(f"FSDP DCP save failed; falling back to HF: {e}")
        # Fall back to HF save
        try:
            trainer.save_model(ckpt_dir)
        except Exception as e2:
            logger.error_rank0(f"HF fallback save also failed: {e2}")

    # Synchronize after save
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass


def fsdp_dcp_load(trainer, ckpt_dir: str) -> None:
    """Load checkpoint via PyTorch Distributed Checkpoint APIs.

    - Loads model using sharded DCP state dict and sets it on the wrapped model.
    - Restores optimizer and scheduler when present; warns clearly if only the model was saved.
    - Uses the data-parallel process group when available; syncs before and after.
    """
    # Synchronize before load
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass

    try:
        from torch.distributed import checkpoint as dist_cp
        from torch.distributed.checkpoint.filesystem import FileSystemReader

        # Version‑compatible DCP imports
        try:  # Prefer new model helpers
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict as dcp_get_model_state_dict,
            )
            from torch.distributed.checkpoint.state_dict import (
                set_model_state_dict as dcp_set_model_state_dict,
            )
        except Exception:
            from torch.distributed.checkpoint.state_dict import (
                get_state_dict as dcp_get_model_state_dict,  # type: ignore
            )
            from torch.distributed.checkpoint.state_dict import (
                set_state_dict as dcp_set_model_state_dict,  # type: ignore
            )

        # Options shims
        try:
            from torch.distributed.checkpoint.state_dict import (
                ModelStateDictOptions as DCPStateDictOptions,  # type: ignore
            )

            def _make_model_options():
                return DCPStateDictOptions(sharded=True)  # type: ignore[arg-type]

        except Exception:
            from torch.distributed.checkpoint.state_dict import (  # type: ignore
                ShardedStateDictConfig,
                StateDictOptions,
            )

            def _make_model_options():
                return StateDictOptions(state_dict_config=ShardedStateDictConfig())

        # Optimizer helpers (may be absent)
        try:
            from torch.distributed.checkpoint.state_dict import (
                get_optimizer_state_dict as dcp_get_optim_sd,  # type: ignore
            )
            from torch.distributed.checkpoint.state_dict import (
                set_optimizer_state_dict as dcp_set_optim_sd,  # type: ignore
            )
        except Exception:
            dcp_get_optim_sd = None  # type: ignore
            dcp_set_optim_sd = None  # type: ignore

        model = trainer.model
        optimizer = getattr(trainer, "optimizer", None)
        scheduler = getattr(trainer, "lr_scheduler", None)

        # Prepare destination containers
        ms_opts = _make_model_options()
        model_dest = dcp_get_model_state_dict(model, options=ms_opts)

        payload: dict[str, Any] = {"model": model_dest}

        # Optimizer destination container
        can_restore_optimizer = optimizer is not None and not _is_nonstandard_optimizer(optimizer)
        opt_dest: Any | None = None
        if can_restore_optimizer:
            try:
                if dcp_get_optim_sd is not None:
                    opt_dest = dcp_get_optim_sd(optimizer)
                    payload["optimizer"] = opt_dest
                else:
                    raise RuntimeError("DCP optimizer helpers unavailable")
            except Exception:
                # Fallback container uses normal state dict structure
                try:
                    opt_dest = optimizer.state_dict()
                    payload["optimizer"] = opt_dest
                except Exception:
                    opt_dest = None

        # Scheduler destination container
        sched_dest: Any | None = None
        if scheduler is not None:
            try:
                sched_dest = scheduler.state_dict()
                payload["scheduler"] = sched_dest
            except Exception:
                sched_dest = None

        reader = FileSystemReader(ckpt_dir)
        process_group = _get_dp_process_group()

        # Perform load into destination containers
        dist_cp.load(payload, storage_reader=reader, process_group=process_group)

        # Set model from loaded state
        dcp_set_model_state_dict(model, payload["model"], options=ms_opts)

        # Restore optimizer if available in checkpoint
        if "optimizer" in payload and opt_dest is not None and optimizer is not None:
            try:
                if dcp_set_optim_sd is not None:
                    dcp_set_optim_sd(optimizer, opt_dest)
                    logger.info_rank0("Restored optimizer state via DCP optimizer API.")
                else:
                    raise RuntimeError("DCP optimizer helpers unavailable")
            except Exception:
                try:
                    optimizer.load_state_dict(opt_dest)
                    logger.info_rank0("Restored optimizer state via standard load_state_dict().")
                except Exception as e:
                    logger.warning_rank0(f"Failed to restore optimizer state; continuing with model-only resume: {e}")
        else:
            logger.warning_rank0(
                "Optimizer state not found in checkpoint or unsupported optimizer detected. Continuing with model-only resume."
            )

        # Restore scheduler if present
        if "scheduler" in payload and sched_dest is not None and scheduler is not None:
            try:
                scheduler.load_state_dict(sched_dest)
            except Exception as e:
                logger.warning_rank0(f"Failed to restore scheduler state; continuing without scheduler resume: {e}")
        elif scheduler is not None:
            logger.warning_rank0("Scheduler state not found in checkpoint. Continuing without scheduler resume.")

        logger.info_rank0(f"Loaded FSDP DCP checkpoint from {ckpt_dir}")

    except Exception as e:
        logger.warning_rank0(f"FSDP DCP load failed; falling back to HF load semantics: {e}")
        # No fallback behavior here; HF Trainer will try its own path if we re-raise/return

    # Synchronize after load
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass
