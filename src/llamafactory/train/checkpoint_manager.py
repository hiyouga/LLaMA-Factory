from __future__ import annotations

import os
import warnings
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


def _is_fsdp_module(mod: torch.nn.Module) -> bool:
    try:
        from torch.distributed.fsdp.api import FullyShardedDataParallel as FSDP  # type: ignore

        if isinstance(mod, FSDP):
            return True
    except Exception:
        pass
    # Heuristic attributes present on FSDP wrappers across torch versions
    for attr in ("_fsdp_wrapped_module", "_fsdp_state", "_handles", "sharding_strategy"):
        if hasattr(mod, attr):
            return True
    return False


def _select_dcp_model_root(model: torch.nn.Module) -> torch.nn.Module:
    """Select the appropriate root module for DCP state dict extraction.

    - Prefer the outermost FSDP wrapper when present to avoid nested _flat_param errors.
    - Otherwise return the original model (or its .module if present).
    """
    # Prefer the wrapper itself if it is FSDP; avoid unwrapping .module in that case
    if _is_fsdp_module(model):
        return model

    candidate = getattr(model, "module", model)
    if _is_fsdp_module(candidate):
        return candidate

    # Find the shallowest FSDP submodule by name depth
    shallow_name = None
    shallow_mod = None
    for name, mod in candidate.named_modules():
        if _is_fsdp_module(mod):
            if shallow_name is None or name.count(".") < shallow_name.count("."):
                shallow_name = name
                shallow_mod = mod
    if shallow_mod is not None:
        return shallow_mod

    return candidate


def fsdp_dcp_save(trainer, output_dir: str) -> None:
    """Save checkpoint via PyTorch Distributed Checkpoint APIs.

    - Saves model via sharded DCP state dict.
    - Saves optimizer only via DCP optimizer helpers when available. Skips non-standard optimizers
      (paged/8-bit/bitsandbytes/DeepSpeed) and does not fall back to optimizer.state_dict(). If args.save_only_model
      is True, optimizer is skipped regardless.
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

        # Options shims (multiple torch variants); prefer DTensor when supported
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
                StateDictType,
            )

            def _make_model_options():
                # Prefer DTensor if available in this torch version
                try:
                    return StateDictOptions(state_dict_config=ShardedStateDictConfig(use_dtensor=True))  # type: ignore[arg-type]
                except TypeError:
                    pass
                try:
                    return StateDictOptions(state_dict_config=ShardedStateDictConfig())  # type: ignore[arg-type]
                except TypeError:
                    pass
                try:
                    return StateDictOptions(model_state_dict_config=ShardedStateDictConfig())  # type: ignore[arg-type]
                except TypeError:
                    pass
                try:
                    return StateDictOptions(state_dict_type=StateDictType.SHARDED_STATE_DICT)  # type: ignore[arg-type]
                except TypeError:
                    pass
                return None

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

        # Build model state dict (sharded); select DCP root and silence ST deprecation
        ms_opts = _make_model_options()
        dcp_model = _select_dcp_model_root(model)
        submods = {dcp_model}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Please use DTensor instead and we are deprecating ShardedTensor.",
                category=FutureWarning,
            )
            if ms_opts is not None:
                model_sd = dcp_get_model_state_dict(dcp_model, submodules=submods, options=ms_opts)
            else:
                model_sd = dcp_get_model_state_dict(dcp_model, submodules=submods)

        payload: dict[str, Any] = {"model": model_sd}

        # Optimizer state (optional)
        include_opt = (not save_only_model) and optimizer is not None
        if include_opt:
            # Also respect the configured optim name if present
            optim_name = getattr(getattr(trainer, "args", None), "optim", None)
            configured_nonstandard = isinstance(optim_name, str) and any(
                x in optim_name.lower() for x in ("8bit", "8_bit", "paged", "lomo", "deepspeed")
            )
            if configured_nonstandard or _is_nonstandard_optimizer(optimizer):
                logger.warning_rank0(
                    "Detected non-standard optimizer (paged/8-bit/bitsandbytes/DeepSpeed). "
                    "Skipping optimizer state save; model weights will still be saved."
                )
            else:
                # Save only via DCP optimizer helpers; do not fall back to .state_dict()
                if dcp_get_optim_sd is not None:
                    try:
                        opt_sd = dcp_get_optim_sd(optimizer)
                        payload["optimizer"] = opt_sd
                        logger.info_rank0("Saving optimizer state via DCP optimizer API.")
                    except Exception as e:
                        logger.warning_rank0(
                            f"Failed to obtain optimizer state via DCP optimizer API: {e}. Skipping optimizer save."
                        )
                else:
                    logger.warning_rank0(
                        "DCP optimizer helpers are unavailable in this torch version. Skipping optimizer state save."
                    )

        # Scheduler state (optional, lightweight)
        include_sched = (not save_only_model) and (scheduler is not None)
        if include_sched:
            try:
                payload["scheduler"] = scheduler.state_dict()
            except Exception as e:
                logger.warning_rank0(f"Failed to get scheduler state_dict; skipping scheduler save: {e}")
        elif scheduler is not None and save_only_model:
            logger.warning_rank0(
                "save_only_model=True; skipping scheduler state save. Model weights will still be saved."
            )

        if optimizer is not None and save_only_model:
            logger.warning_rank0(
                "save_only_model=True; skipping optimizer state save. Model weights will still be saved."
            )

        writer = FileSystemWriter(ckpt_dir)

        # Use DP group if available; else default
        process_group = _get_dp_process_group()

        dist_cp.save(payload, storage_writer=writer, process_group=process_group)
        logger.info_rank0(f"Saved FSDP DCP checkpoint to {ckpt_dir}")

        # Synchronize after successful save
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception:
            pass

    except Exception as e:
        # Do not fall back to HF save here; fail fast to surface DCP issues explicitly
        logger.warning_rank0(f"FSDP DCP save failed: {e}")
        raise


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

        # Options shims (multiple torch variants); prefer DTensor when supported
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
                StateDictType,
            )

            def _make_model_options():
                # Prefer DTensor if available in this torch version
                try:
                    return StateDictOptions(state_dict_config=ShardedStateDictConfig(use_dtensor=True))  # type: ignore[arg-type]
                except TypeError:
                    pass
                try:
                    return StateDictOptions(state_dict_config=ShardedStateDictConfig())  # type: ignore[arg-type]
                except TypeError:
                    pass
                try:
                    return StateDictOptions(model_state_dict_config=ShardedStateDictConfig())  # type: ignore[arg-type]
                except TypeError:
                    pass
                try:
                    return StateDictOptions(state_dict_type=StateDictType.SHARDED_STATE_DICT)  # type: ignore[arg-type]
                except TypeError:
                    pass
                return None

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
        dcp_model = _select_dcp_model_root(model)
        submods = {dcp_model}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Please use DTensor instead and we are deprecating ShardedTensor.",
                category=FutureWarning,
            )
            if ms_opts is not None:
                model_dest = dcp_get_model_state_dict(dcp_model, submodules=submods, options=ms_opts)
            else:
                model_dest = dcp_get_model_state_dict(dcp_model, submodules=submods)

        payload: dict[str, Any] = {"model": model_dest}

        # Optimizer destination container
        optim_name = getattr(getattr(trainer, "args", None), "optim", None)
        configured_nonstandard = isinstance(optim_name, str) and any(
            x in optim_name.lower() for x in ("8bit", "8_bit", "paged", "lomo", "deepspeed")
        )
        is_nonstandard_opt = (
            (_is_nonstandard_optimizer(optimizer) or configured_nonstandard) if optimizer is not None else False
        )
        can_restore_optimizer = optimizer is not None and not is_nonstandard_opt
        if is_nonstandard_opt:
            logger.warning_rank0(
                "Detected non-standard optimizer (paged/8-bit/bitsandbytes/DeepSpeed). "
                "Will not attempt optimizer state restore; proceeding with model-only resume."
            )
        opt_dest: Any | None = None
        if can_restore_optimizer:
            try:
                if dcp_get_optim_sd is not None:
                    opt_dest = dcp_get_optim_sd(optimizer)
                    payload["optimizer"] = opt_dest
                else:
                    raise RuntimeError("DCP optimizer helpers unavailable")
            except Exception:
                # If we cannot build a DCP optimizer container, skip optimizer restore
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
        if ms_opts is not None:
            dcp_set_model_state_dict(dcp_model, payload["model"], options=ms_opts)
        else:
            dcp_set_model_state_dict(dcp_model, payload["model"])  # type: ignore[misc]

        # Restore optimizer if available in checkpoint
        if "optimizer" in payload and opt_dest is not None and optimizer is not None:
            try:
                if dcp_set_optim_sd is not None:
                    dcp_set_optim_sd(optimizer, opt_dest)
                    logger.info_rank0("Restored optimizer state via DCP optimizer API.")
                else:
                    raise RuntimeError("DCP optimizer helpers unavailable")
            except Exception:
                logger.warning_rank0("Failed to restore optimizer via DCP API; continuing with model-only resume.")
        else:
            logger.warning_rank0(
                "Optimizer state not restored (either unsupported optimizer or not present). "
                "Continuing with model-only resume."
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
