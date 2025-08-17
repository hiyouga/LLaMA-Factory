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

"""Debugging utilities for distributed training and tensor issues."""

from typing import Any, Dict

import torch
import torch.distributed as dist

from ..extras.logging import get_logger


logger = get_logger(__name__)


def debug_tensor_properties(tensor: torch.Tensor, name: str = "tensor", rank_filter: int = None) -> None:
    """Debug tensor properties with distributed context.
    
    Args:
        tensor: The tensor to debug
        name: Name/identifier for the tensor
        rank_filter: Only log from this specific rank (None = all ranks)
    """
    if rank_filter is not None and dist.is_initialized() and dist.get_rank() != rank_filter:
        return

    rank_str = f"[rank{dist.get_rank()}]" if dist.is_initialized() else "[single]"

    logger.info_rank0(f"{rank_str} Tensor '{name}' - "
                     f"shape: {tensor.shape}, "
                     f"dtype: {tensor.dtype}, "
                     f"device: {tensor.device}, "
                     f"is_contiguous: {tensor.is_contiguous()}, "
                     f"is_cuda: {tensor.is_cuda}, "
                     f"requires_grad: {tensor.requires_grad}")

    # Check for common issues
    if not tensor.is_contiguous():
        logger.warning(f"{rank_str} Non-contiguous tensor '{name}' - this may cause distributed ops to fail")

    if tensor.device.type == 'cpu' and dist.is_initialized():
        logger.warning(f"{rank_str} CPU tensor '{name}' in distributed context - may need to be on CUDA")


def debug_batch_properties(batch: Dict[str, Any], batch_idx: int = None, rank_filter: int = 0) -> None:
    """Debug all tensors in a batch.
    
    Args:
        batch: Dictionary containing batch data
        batch_idx: Optional batch index for identification
        rank_filter: Only log from this specific rank (0 = rank 0 only)
    """
    if rank_filter is not None and dist.is_initialized() and dist.get_rank() != rank_filter:
        return

    rank_str = f"[rank{dist.get_rank()}]" if dist.is_initialized() else "[single]"
    batch_str = f"batch_{batch_idx}" if batch_idx is not None else "batch"

    logger.info_rank0(f"{rank_str} === Debugging {batch_str} ===")

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            debug_tensor_properties(value, f"{batch_str}.{key}", rank_filter=None)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            for i, tensor in enumerate(value[:3]):  # Debug first 3 items
                debug_tensor_properties(tensor, f"{batch_str}.{key}[{i}]", rank_filter=None)
        else:
            logger.info_rank0(f"{rank_str} {batch_str}.{key}: {type(value)} (not tensor)")


def validate_cuda_tensors(batch: Dict[str, torch.Tensor], operation_name: str = "operation") -> bool:
    """Validate that all tensors are CUDA and dense before distributed operations.
    
    Args:
        batch: Dictionary of tensors to validate
        operation_name: Name of the operation for error messages
    
    Returns:
        True if all tensors are valid, False otherwise
    """
    rank_str = f"[rank{dist.get_rank()}]" if dist.is_initialized() else "[single]"
    valid = True

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if not value.is_cuda and dist.is_initialized():
                logger.error(f"{rank_str} {operation_name}: Tensor '{key}' is not on CUDA (device: {value.device})")
                valid = False

            if not value.is_contiguous():
                logger.error(f"{rank_str} {operation_name}: Tensor '{key}' is not contiguous")
                valid = False

            # Check for sparse tensors
            if value.is_sparse:
                logger.error(f"{rank_str} {operation_name}: Tensor '{key}' is sparse (needs dense)")
                valid = False

    return valid


def debug_gather_operation(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Debug a tensor before a gather operation.
    
    Args:
        tensor: Tensor that will be used in gather operation
        name: Name for identification
    """
    rank_str = f"[rank{dist.get_rank()}]" if dist.is_initialized() else "[single]"

    logger.info_rank0(f"{rank_str} Pre-gather debug '{name}': "
                     f"shape={tensor.shape}, dtype={tensor.dtype}, "
                     f"device={tensor.device}, is_cuda={tensor.is_cuda}, "
                     f"is_contiguous={tensor.is_contiguous()}, "
                     f"is_sparse={tensor.is_sparse}")

    # Validate for gather operation
    if not tensor.is_cuda and dist.is_initialized():
        logger.error(f"{rank_str} GATHER ERROR: '{name}' must be on CUDA for distributed gather")

    if not tensor.is_contiguous():
        logger.error(f"{rank_str} GATHER ERROR: '{name}' must be contiguous for distributed gather")

    if tensor.is_sparse:
        logger.error(f"{rank_str} GATHER ERROR: '{name}' must be dense (not sparse) for distributed gather")


# Quick enable/disable debug flags
DEBUG_TENSORS = False  # Set to True to enable tensor debugging
DEBUG_BATCHES = False  # Set to True to enable batch debugging
DEBUG_GATHER = False   # Set to True to enable gather operation debugging
