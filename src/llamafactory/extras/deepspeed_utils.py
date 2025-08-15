# Copyright 2025 - LLaMA Factory Contributors
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

"""Utilities for working with DeepSpeed and fixing known issues."""

import functools
from typing import Any

from ..extras.logging import get_logger


logger = get_logger(__name__)


def patch_deepspeed_pin_memory_bug() -> None:
    """Monkey patch DeepSpeed to fix the pin_memory bug that causes crashes.

    This addresses a known issue in DeepSpeed where it doesn't respect
    pin_memory=False settings and always tries to pin memory, causing
    "CUDA error: invalid argument" crashes in certain environments.

    The patch wraps the accelerator's pin_memory function to gracefully
    handle failures and return the original tensor unpinned when pinning fails.
    """
    try:
        from deepspeed import get_accelerator
    except ImportError:
        logger.debug("DeepSpeed not available, skipping pin_memory patch")
        return

    # Get the accelerator and check if it has pin_memory
    accelerator = get_accelerator()
    if not hasattr(accelerator, "pin_memory"):
        logger.debug("DeepSpeed accelerator has no pin_memory method, skipping patch")
        return

    # Store the original function
    original_pin_memory = accelerator.pin_memory

    # Check if already patched
    if hasattr(original_pin_memory, "_llamafactory_patched"):
        logger.debug("DeepSpeed pin_memory already patched")
        return

    @functools.wraps(original_pin_memory)
    def patched_pin_memory(tensor: Any, align_bytes: int = 1) -> Any:
        """Patched version of pin_memory that handles CUDA errors gracefully.

        Args:
            tensor: The tensor to pin in memory
            align_bytes: Memory alignment bytes

        Returns:
            The pinned tensor if successful, otherwise a CPU contiguous tensor
        """
        try:
            return original_pin_memory(tensor, align_bytes)
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "invalid argument" in error_msg or "cuda error" in error_msg:
                logger.warning(
                    f"DeepSpeed pin_memory failed with error: {e}. "
                    "Returning CPU contiguous tensor. Consider setting pin_memory=False "
                    "in your training configuration."
                )
                # Ensure we return a CPU tensor that's contiguous and has the right properties
                try:
                    if hasattr(tensor, "cpu") and hasattr(tensor, "contiguous"):
                        # PyTorch tensor - move to CPU and make contiguous
                        cpu_tensor = tensor.cpu().contiguous()
                        return cpu_tensor
                    else:
                        # Fallback for non-PyTorch tensors
                        return tensor
                except Exception as fallback_error:
                    logger.warning(
                        f"Failed to create CPU contiguous tensor: {fallback_error}. Returning original tensor."
                    )
                    return tensor
            # Re-raise other RuntimeErrors
            raise
        except Exception as e:
            logger.warning(
                f"DeepSpeed pin_memory failed with unexpected error: {e}. Returning CPU contiguous tensor if possible."
            )
            # Same fallback logic for other exceptions
            try:
                if hasattr(tensor, "cpu") and hasattr(tensor, "contiguous"):
                    cpu_tensor = tensor.cpu().contiguous()
                    return cpu_tensor
                else:
                    return tensor
            except Exception:
                return tensor

    # Mark the function as patched to avoid double-patching
    patched_pin_memory._llamafactory_patched = True  # type: ignore

    # Apply the patch
    accelerator.pin_memory = patched_pin_memory

    logger.info("Applied DeepSpeed pin_memory patch to handle CUDA pinning failures gracefully")


def create_cache_clearing_callback() -> Any:
    """Create a training callback that periodically clears GPU cache.

    This helps prevent the DeepSpeed warning about cache flushes
    by proactively clearing cache at regular intervals.

    Returns:
        A TrainerCallback for cache clearing
    """
    try:
        import torch
        from transformers import TrainerCallback
    except ImportError:
        logger.warning("Cannot create cache clearing callback - missing dependencies")
        return None

    class CacheClearingCallback(TrainerCallback):
        """Callback to periodically clear GPU cache during training."""

        def __init__(self, clear_every_n_steps: int = 10):
            self.clear_every_n_steps = clear_every_n_steps

        def on_step_end(self, args, state, control, **kwargs):
            """Clear cache every N steps."""
            if state.global_step > 0 and state.global_step % self.clear_every_n_steps == 0:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to clear device cache: {e}")

    return CacheClearingCallback


def apply_deepspeed_patches() -> None:
    """Apply all known DeepSpeed patches/fixes."""
    patch_deepspeed_pin_memory_bug()
