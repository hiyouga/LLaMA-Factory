from __future__ import annotations

import warnings

from . import logging


logger = logging.get_logger(__name__)


def apply_compile_shims(enable_liger: bool, torch_compile: bool, enable_deepspeed: bool = False) -> None:
    """Apply small runtime shims to reduce TorchDynamo/DeepSpeed issues under compile.

    - Liger + compile: mark known Liger Python entrypoints as non-traceable (dynamo.disable)
      and suppress noisy pybind warnings.
    - DeepSpeed + compile: replace CUDA event timers with CPU timers to avoid invalid CUDA
      event usage in DS timing/logging paths.
    """
    if not torch_compile:
        return

    try:
        import torch._dynamo as dynamo  # type: ignore

        if enable_liger:
            try:
                # Liger: fused CE Python wrappers
                import liger_kernel.transformers.model.loss_utils as loss_utils  # type: ignore

                if hasattr(loss_utils, "fixed_fused_linear_cross_entropy"):
                    loss_utils.fixed_fused_linear_cross_entropy = dynamo.disable(  # type: ignore[attr-defined]
                        loss_utils.fixed_fused_linear_cross_entropy
                    )
                    logger.info_rank0("Marked Liger fixed_fused_linear_cross_entropy as non-traceable for Dynamo.")

                if hasattr(loss_utils, "LigerForCausalLMLoss"):
                    loss_utils.LigerForCausalLMLoss = dynamo.disable(  # type: ignore[attr-defined]
                        loss_utils.LigerForCausalLMLoss
                    )
                    logger.info_rank0("Marked Liger LigerForCausalLMLoss as non-traceable for Dynamo.")
            except Exception:
                pass

            # Liger: RMSNorm custom autograd and module forward
            try:
                import liger_kernel.transformers.rms_norm as rms_norm  # type: ignore

                if hasattr(rms_norm, "LigerRMSNormFunction") and hasattr(rms_norm.LigerRMSNormFunction, "apply"):
                    rms_norm.LigerRMSNormFunction.apply = dynamo.disable(  # type: ignore[attr-defined]
                        rms_norm.LigerRMSNormFunction.apply
                    )
                    logger.info_rank0("Marked Liger LigerRMSNormFunction.apply as non-traceable for Dynamo.")

                if hasattr(rms_norm, "LigerRMSNorm") and hasattr(rms_norm.LigerRMSNorm, "forward"):
                    rms_norm.LigerRMSNorm.forward = dynamo.disable(  # type: ignore[attr-defined]
                        rms_norm.LigerRMSNorm.forward
                    )
                    logger.info_rank0("Marked Liger LigerRMSNorm.forward as non-traceable for Dynamo.")
            except Exception:
                pass

        # Best-effort: reduce noise from pybind11_object.__new__ warnings
        try:
            warnings.filterwarnings(
                "ignore",
                message=r".*pybind11_object.__new__.*",
                module=r"torch\._dynamo\..*",
            )
            logger.debug_rank0("Suppressed Dynamo pybind11_object warnings.")
        except Exception:
            pass
    except Exception:
        # If dynamo not available or import fails, do nothing
        pass

    # DeepSpeed timing shim: replace CUDA event timers with CPU timers under compile
    if enable_deepspeed:
        try:
            import deepspeed.utils.timer as ds_timer  # type: ignore

            if hasattr(ds_timer, "EventTimer") and hasattr(ds_timer, "CPUTimer"):
                ds_timer.EventTimer = ds_timer.CPUTimer  # type: ignore[attr-defined]
                logger.info_rank0(
                    "Disabled DeepSpeed CUDA event timers under torch.compile; using CPU timing (approximate)."
                )
        except Exception:
            # Fallback: guard elapsed computation to avoid crashes
            try:
                import deepspeed.utils.timer as ds_timer  # type: ignore

                def _safe_elapsed(self):
                    try:
                        return float(getattr(self, "end_time", 0.0) - getattr(self, "start_time", 0.0))
                    except Exception:
                        return 0.0

                if hasattr(ds_timer, "EventTimer"):
                    setattr(ds_timer.EventTimer, "get_elapsed_msec", _safe_elapsed)  # type: ignore
                    logger.info_rank0("Patched DeepSpeed EventTimer.get_elapsed_msec with a safe CPU fallback.")
            except Exception:
                pass
