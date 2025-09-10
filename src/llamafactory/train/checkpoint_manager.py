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


def _debug_log_model_wrappers(model: torch.nn.Module) -> None:
    try:
        types_chain = [type(model).__name__]
        probe = model
        # Walk a short .module chain, which can include wrappers like DDP/compile/PEFT
        for _ in range(4):
            if hasattr(probe, "module") and isinstance(getattr(probe, "module"), torch.nn.Module):
                probe = getattr(probe, "module")
                types_chain.append(type(probe).__name__)
            else:
                break
        logger.info_rank0(f"DCP root discovery: wrapper chain types = {types_chain}")

        # Scan for FSDP modules and find the shallowest path
        shallow = None
        shallow_name = None
        fsdp_count = 0
        sample_paths = []
        for name, mod in probe.named_modules():
            if _is_fsdp_module(mod):
                fsdp_count += 1
                if len(sample_paths) < 5:
                    sample_paths.append(name or "<root>")
                if shallow_name is None or name.count(".") < shallow_name.count("."):
                    shallow = mod
                    shallow_name = name
        logger.info_rank0(f"DCP root discovery: FSDP modules detected = {fsdp_count}; sample paths = {sample_paths}")
        if shallow is not None:
            logger.info_rank0(
                f"DCP root discovery: shallowest FSDP path = '{shallow_name or '<root>'}', type = {type(shallow).__name__}"
            )
        else:
            logger.info_rank0("DCP root discovery: no FSDP modules found under unwrapped model")
    except Exception as e:
        logger.info_rank0(f"DCP root discovery: logging failed: {e}")


def _select_dcp_model_root(model: torch.nn.Module) -> torch.nn.Module:
    """Select the appropriate root module for DCP state dict.

    - If the top-level object is an FSDP wrapper, return it (covers full model).
    - Else unwrap a short `.module` chain and special-case `OptimizedModule._orig_mod`
      to reach the real base model. Return this base so DCP can traverse all nested
      FSDP wrappers. Avoid selecting an individual FSDP submodule, which can lead to
      nested _flat_param errors and partial saves.
    """
    _debug_log_model_wrappers(model)

    if _is_fsdp_module(model):
        logger.info_rank0("DCP root selection: top-level is FSDP; using model as root")
        return model

    base = model
    for _ in range(6):
        next_mod = getattr(base, "module", None)
        if isinstance(next_mod, torch.nn.Module):
            base = next_mod
            continue
        # torch.compile OptimizedModule exposes original module as _orig_mod
        orig = getattr(base, "_orig_mod", None)
        if isinstance(orig, torch.nn.Module):
            base = orig
            continue
        break

    try:
        logger.info_rank0(f"DCP root selection: using base type = {type(base).__name__}")
    except Exception:
        pass
    return base


_GLOO_DCP_GROUP = None


def _get_gloo_process_group():
    """Create or return a cached Gloo process group for control-plane collectives.

    Using Gloo for DCP planning (gather_object/reduce_scatter) can avoid NCCL GPU memory
    pressure and reduce the chance of GPU OOM during checkpoint save/load.
    """
    global _GLOO_DCP_GROUP
    try:
        if not torch.distributed.is_initialized():
            return None
        if _GLOO_DCP_GROUP is not None:
            return _GLOO_DCP_GROUP
        _GLOO_DCP_GROUP = torch.distributed.new_group(backend="gloo")
        return _GLOO_DCP_GROUP
    except Exception:
        return None


# -- DCP Stateful wrapper ----------------------------------------------------
try:
    from torch.distributed.checkpoint.state_dict import get_state_dict as dcp_get_state_dict  # type: ignore
    from torch.distributed.checkpoint.state_dict import set_state_dict as dcp_set_state_dict  # type: ignore

    # Prefer new model-only helpers when available
    try:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict as dcp_get_model_state_dict,  # type: ignore
        )
    except Exception:
        dcp_get_model_state_dict = None  # type: ignore
    try:
        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict as dcp_set_model_state_dict,  # type: ignore
        )
    except Exception:
        dcp_set_model_state_dict = None  # type: ignore
    from torch.distributed.checkpoint.stateful import Stateful

    # Optional options API for better nested-FSDP handling
    try:
        from torch.distributed.checkpoint.state_dict import (
            ModelStateDictOptions as DCPModelStateDictOptions,  # type: ignore
        )
    except Exception:
        DCPModelStateDictOptions = None  # type: ignore
    try:
        from torch.distributed.checkpoint.state_dict import (  # type: ignore
            ShardedStateDictConfig,
            StateDictOptions,
            StateDictType,
        )
    except Exception:
        ShardedStateDictConfig = None  # type: ignore
        StateDictOptions = None  # type: ignore
        StateDictType = None  # type: ignore
except Exception:  # pragma: no cover - older torch variants unsupported for Stateful path
    Stateful = object  # type: ignore
    dcp_get_state_dict = None  # type: ignore
    dcp_set_state_dict = None  # type: ignore
    dcp_get_model_state_dict = None  # type: ignore
    dcp_set_model_state_dict = None  # type: ignore
    DCPModelStateDictOptions = None  # type: ignore
    ShardedStateDictConfig = None  # type: ignore
    StateDictOptions = None  # type: ignore
    StateDictType = None  # type: ignore


def _make_dcp_model_options():
    """Create DCP model state dict options with sharded/DTensor preference across torch versions."""
    # Newer API
    if DCPModelStateDictOptions is not None:
        try:
            return DCPModelStateDictOptions(sharded=True)  # type: ignore[arg-type]
        except Exception:
            pass
    # Older API variants
    if StateDictOptions is not None and ShardedStateDictConfig is not None:
        try:
            return StateDictOptions(state_dict_config=ShardedStateDictConfig(use_dtensor=True))  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            return StateDictOptions(state_dict_config=ShardedStateDictConfig())  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            return StateDictOptions(model_state_dict_config=ShardedStateDictConfig())  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            return StateDictOptions(state_dict_type=StateDictType.SHARDED_STATE_DICT)  # type: ignore[arg-type]
        except Exception:
            pass
    return None


class AppState(Stateful):  # type: ignore[misc]
    """Stateful wrapper aligning with PyTorch DCP tutorial.

    This class centralizes state generation and restoration for model/optimizer
    and optionally scheduler. When used with DCP's save/load, DCP will call
    state_dict()/load_state_dict() automatically.
    """

    def __init__(self, model: torch.nn.Module, optimizer: Any | None = None, scheduler: Any | None = None):
        # Prefer the outermost FSDP wrapper (or shallowest FSDP module)
        self.model = _select_dcp_model_root(model)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        state: dict[str, Any] = {}
        # Default to FSDP.SHARDED_STATE_DICT and handle FQNs
        if dcp_get_state_dict is not None:
            opts = _make_dcp_model_options()
            if self.optimizer is not None:
                try:
                    model_sd, optim_sd = dcp_get_state_dict(self.model, self.optimizer, options=opts)  # type: ignore[misc]
                except TypeError:
                    # Some versions require positional options or no options
                    model_sd, optim_sd = dcp_get_state_dict(self.model, self.optimizer)  # type: ignore[misc]
                try:
                    logger.info_rank0("DCP save path: model+optimizer via get_state_dict")
                except Exception:
                    pass
                state["model"] = model_sd
                state["optim"] = optim_sd
            else:
                # Model-only path: prefer dedicated model helper when available
                try:
                    if dcp_get_model_state_dict is not None:
                        try:
                            model_sd = dcp_get_model_state_dict(self.model, options=opts)  # type: ignore[misc]
                        except TypeError:
                            model_sd = dcp_get_model_state_dict(self.model)  # type: ignore[misc]
                        try:
                            logger.info_rank0("DCP save path: model-only via get_model_state_dict")
                        except Exception:
                            pass
                    else:
                        # Fallback: explicit empty optimizers iterable
                        try:
                            model_sd, _ = dcp_get_state_dict(self.model, (), options=opts)  # type: ignore[misc]
                        except TypeError:
                            model_sd, _ = dcp_get_state_dict(self.model, ())  # type: ignore[misc]
                        try:
                            logger.info_rank0("DCP save path: model-only via get_state_dict(empty)")
                        except Exception:
                            pass
                    state["model"] = model_sd
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "_flat_param contains _flat_param" in msg or "not the root module" in msg:
                        # Fallback: save a forest of outermost FSDP submodules
                        logger.warning_rank0(
                            "DCP model-level save failed due to nested FSDP. Falling back to per-submodule save."
                        )
                        # Build forest of outermost FSDP modules under the selected base
                        fsdp_paths: list[tuple[str, torch.nn.Module]] = []
                        all_paths: dict[str, torch.nn.Module] = {}
                        for name, mod in self.model.named_modules():
                            if _is_fsdp_module(mod):
                                all_paths[name] = mod
                        # Keep only paths that do not have a shorter FSDP ancestor
                        for name, mod in all_paths.items():
                            parent = name.rsplit(".", 1)[0] if "." in name else ""
                            has_fsdp_parent = False
                            while parent:
                                if parent in all_paths:
                                    has_fsdp_parent = True
                                    break
                                parent = parent.rsplit(".", 1)[0] if "." in parent else ""
                            if not has_fsdp_parent:
                                fsdp_paths.append((name, mod))
                        forest_sd: dict[str, Any] = {"__fsdp_forest__": True}
                        for name, mod in fsdp_paths:
                            try:
                                part_sd = (
                                    dcp_get_model_state_dict(mod, options=opts)  # type: ignore[misc]
                                    if dcp_get_model_state_dict is not None
                                    else dcp_get_state_dict(mod)[0]  # type: ignore[misc]
                                )
                                forest_sd[name] = part_sd
                            except Exception as e2:
                                logger.warning_rank0(f"Skipping FSDP submodule '{name}' due to: {e2}")
                        try:
                            sample = [n for n in forest_sd.keys() if not n.startswith("__")][:5]
                            logger.info_rank0(
                                f"DCP save path: per-submodule forest, roots={len(forest_sd)-1}, sample_paths={sample}"
                            )
                        except Exception:
                            pass
                        if len(forest_sd) <= 1:
                            raise
                        state["model"] = forest_sd
                    else:
                        raise
        else:
            # Extremely old torch: fall back to raw containers
            state["model"] = getattr(self.model, "state_dict")()
            if self.optimizer is not None:
                state["optim"] = self.optimizer.state_dict()
        # Lightweight scheduler (not managed by DCP helpers)
        if self.scheduler is not None:
            try:
                state["scheduler"] = self.scheduler.state_dict()
            except Exception:
                pass
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        # Restore model/optimizer via DCP helpers when available
        if dcp_set_state_dict is not None:
            try:
                opts = _make_dcp_model_options()
                if self.optimizer is not None:
                    try:
                        dcp_set_state_dict(
                            self.model,
                            self.optimizer,
                            model_state_dict=state_dict.get("model"),
                            optim_state_dict=state_dict.get("optim"),
                            options=opts,  # type: ignore[misc]
                        )
                    except TypeError:
                        dcp_set_state_dict(
                            self.model,
                            self.optimizer,
                            model_state_dict=state_dict.get("model"),
                            optim_state_dict=state_dict.get("optim"),
                        )
                    try:
                        logger.info_rank0("DCP load path: model+optimizer via set_state_dict")
                    except Exception:
                        pass
                else:
                    # Handle per-submodule forest payload
                    payload_model = state_dict.get("model")
                    if isinstance(payload_model, dict) and payload_model.get("__fsdp_forest__"):
                        logger.info_rank0("Restoring FSDP per-submodule forest payload.")
                        for name, part in payload_model.items():
                            if name.startswith("__"):
                                continue
                            # Resolve submodule by dotted name under self.model
                            mod: torch.nn.Module = self.model
                            try:
                                for seg in name.split(".") if name else []:
                                    if not seg:
                                        continue
                                    mod = getattr(mod, seg)
                            except Exception:
                                logger.warning_rank0(f"Cannot resolve submodule '{name}' for DCP restore; skipping.")
                                continue
                            if dcp_set_model_state_dict is not None:
                                try:
                                    dcp_set_model_state_dict(mod, part, options=opts)  # type: ignore[misc]
                                except TypeError:
                                    dcp_set_model_state_dict(mod, part)  # type: ignore[misc]
                            else:
                                try:
                                    dcp_set_state_dict(
                                        mod, (), model_state_dict=part, optim_state_dict=None, options=opts
                                    )  # type: ignore[misc]
                                except TypeError:
                                    dcp_set_state_dict(mod, (), model_state_dict=part, optim_state_dict=None)  # type: ignore[misc]
                        try:
                            logger.info_rank0(
                                f"DCP load path: per-submodule forest, parts={len([k for k in payload_model.keys() if not k.startswith('__')])}"
                            )
                        except Exception:
                            pass
                    else:
                        # Prefer dedicated model helper
                        if dcp_set_model_state_dict is not None:
                            try:
                                dcp_set_model_state_dict(self.model, payload_model, options=opts)  # type: ignore[misc]
                            except TypeError:
                                dcp_set_model_state_dict(self.model, payload_model)  # type: ignore[misc]
                        else:
                            try:
                                dcp_set_state_dict(
                                    self.model,
                                    (),  # explicit empty iterable for optimizers
                                    model_state_dict=payload_model,
                                    optim_state_dict=None,
                                    options=opts,  # type: ignore[misc]
                                )
                            except TypeError:
                                dcp_set_state_dict(
                                    self.model,
                                    (),
                                    model_state_dict=payload_model,
                                    optim_state_dict=None,
                                )
                        try:
                            logger.info_rank0("DCP load path: model-only via set_model_state_dict/set_state_dict")
                        except Exception:
                            pass
            except Exception:
                pass
        # Scheduler restore (best-effort)
        if self.scheduler is not None and "scheduler" in state_dict:
            try:
                self.scheduler.load_state_dict(state_dict["scheduler"])  # type: ignore[attr-defined]
            except Exception:
                pass


def fsdp_dcp_save(trainer, output_dir: str) -> None:
    """Save checkpoint via PyTorch Distributed Checkpoint APIs (Stateful style).

    - Uses a Stateful `AppState` to generate sharded model/optimizer state dicts in line
      with the official DCP tutorial, automatically handling FSDP FQNs and defaults.
    - Skips non-standard optimizers (paged/8-bit/bitsandbytes/DeepSpeed). Respects `save_only_model`.
    - Saves scheduler via .state_dict() when provided and not `save_only_model`.
    - Uses the data-parallel process group when available; includes Gloo fallback on NCCL errors.
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
        import torch.distributed.checkpoint as dcp

        model = trainer.model
        optimizer = getattr(trainer, "optimizer", None)
        scheduler = getattr(trainer, "lr_scheduler", None)
        save_only_model = getattr(trainer.args, "save_only_model", False)

        # Decide optimizer inclusion
        include_opt = (not save_only_model) and (optimizer is not None)
        optim_name = getattr(getattr(trainer, "args", None), "optim", None)
        configured_nonstandard = isinstance(optim_name, str) and any(
            x in optim_name.lower() for x in ("8bit", "8_bit", "paged", "lomo", "deepspeed")
        )
        if include_opt and (configured_nonstandard or _is_nonstandard_optimizer(optimizer)):  # type: ignore[arg-type]
            logger.warning_rank0(
                "Detected non-standard optimizer (paged/8-bit/bitsandbytes/DeepSpeed). "
                "Skipping optimizer state save; model weights will still be saved."
            )
            include_opt = False

        include_sched = (not save_only_model) and (scheduler is not None)
        if not include_sched and scheduler is not None:
            logger.warning_rank0(
                "save_only_model=True; skipping scheduler state save. Model weights will still be saved."
            )
        if not include_opt and optimizer is not None and save_only_model:
            logger.warning_rank0(
                "save_only_model=True; skipping optimizer state save. Model weights will still be saved."
            )

        app = AppState(model, optimizer if include_opt else None, scheduler if include_sched else None)
        state: dict[str, Any] = {"app": app}

        # Use DP group if available; else default
        process_group = _get_dp_process_group()

        try:
            # Proactively release cached GPU memory to reduce NCCL/DCP planning pressure
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            dcp.save(state, checkpoint_id=ckpt_dir, process_group=process_group)
            logger.info_rank0(f"Saved FSDP DCP checkpoint to {ckpt_dir}")
        except Exception as e:
            msg = str(e).lower()
            likely_nccl = "nccl" in msg or "unhandled cuda error" in msg
            # Heuristic detection of CUDA OOM/alloc failures
            oom_markers = (
                "out of memory",
                "cuda oom",
                "cuda out of memory",
                "cuda error out of memory",
                "cublas_status_alloc_failed",
                "cudnn_status_alloc_failed",
                "cuda runtime error: out of memory",
                "ncclinternalerror: unhandled cuda error",  # often follows OOM
            )
            is_oom_like = any(m in msg for m in oom_markers)
            # Log a quick GPU memory snapshot to help diagnose hidden OOMs
            try:
                if torch.cuda.is_available():
                    dev = torch.cuda.current_device()
                    free, total = torch.cuda.mem_get_info()
                    alloc = torch.cuda.memory_allocated(dev)
                    rsvd = torch.cuda.memory_reserved(dev)
                    logger.warning_rank0(
                        f"GPU memory snapshot before DCP save retry: free={free/1e9:.2f}GB, total={total/1e9:.2f}GB, "
                        f"allocated={alloc/1e9:.2f}GB, reserved={rsvd/1e9:.2f}GB"
                    )
            except Exception:
                pass
            # Retry with a CPU/Gloo process group to lower GPU memory pressure during planning
            gloo_pg = _get_gloo_process_group()
            if gloo_pg is not None and likely_nccl:
                logger.warning_rank0(
                    "FSDP DCP save encountered a NCCL error; retrying with Gloo process group for checkpointing."
                )
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                dcp.save(state, checkpoint_id=ckpt_dir, process_group=gloo_pg)
                logger.info_rank0(f"Saved FSDP DCP checkpoint to {ckpt_dir} using Gloo fallback")
            else:
                # Only emit the GPU memory pressure/OOM hint when the error looks like OOM
                if is_oom_like:
                    raise RuntimeError(
                        "FSDP DCP save failed, likely due to CUDA OOM or GPU memory pressure. "
                        "Consider enabling NCCL_ASYNC_ERROR_HANDLING=1, retrying with fewer ranks or shorter cutoff_len, "
                        "or using CPU offload. Original error: " + str(e)
                    )
                # Otherwise, bubble up the original error to avoid misleading diagnostics
                raise

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
    """Load checkpoint via PyTorch Distributed Checkpoint APIs (Stateful style).

    - Constructs a Stateful `AppState` and delegates state generation and restoration
      to DCP's get/set_state_dict helpers as per the official tutorial.
    - Skips non-standard optimizer restore (paged/8-bit/bitsandbytes/DeepSpeed).
    - Uses the data-parallel process group when available; includes Gloo fallback on NCCL errors.
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
        import torch.distributed.checkpoint as dcp

        model = trainer.model
        optimizer = getattr(trainer, "optimizer", None)
        scheduler = getattr(trainer, "lr_scheduler", None)

        # Respect non-standard optimizer limitation
        optim_name = getattr(getattr(trainer, "args", None), "optim", None)
        configured_nonstandard = isinstance(optim_name, str) and any(
            x in optim_name.lower() for x in ("8bit", "8_bit", "paged", "lomo", "deepspeed")
        )
        is_nonstandard_opt = optimizer is not None and (_is_nonstandard_optimizer(optimizer) or configured_nonstandard)
        if is_nonstandard_opt:
            logger.warning_rank0(
                "Detected non-standard optimizer (paged/8-bit/bitsandbytes/DeepSpeed). "
                "Will not attempt optimizer state restore; proceeding with model-only resume."
            )

        app = AppState(model, None if is_nonstandard_opt else optimizer, scheduler)
        state: dict[str, Any] = {"app": app}

        process_group = _get_dp_process_group()

        # Perform load via DCP Stateful
        try:
            dcp.load(state_dict=state, checkpoint_id=ckpt_dir, process_group=process_group)
        except Exception as e:
            msg = str(e).lower()
            likely_nccl = "nccl" in msg or "unhandled cuda error" in msg
            gloo_pg = _get_gloo_process_group()
            if gloo_pg is not None and likely_nccl:
                logger.warning_rank0(
                    "FSDP DCP load encountered a NCCL error; retrying with Gloo process group for checkpointing."
                )
                dcp.load(state_dict=state, checkpoint_id=ckpt_dir, process_group=gloo_pg)
            else:
                raise

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
