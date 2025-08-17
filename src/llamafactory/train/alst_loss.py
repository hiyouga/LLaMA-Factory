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

"""ALST (Arctic Long Sequence Training) loss computation utilities.

Based on the ArcticTraining implementation pattern:
https://github.com/snowflakedb/ArcticTraining/blob/main/arctic_training/trainer/sft_trainer.py
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn.functional
from torch.nn import CrossEntropyLoss

from ..extras import logging


if TYPE_CHECKING:
    from torch.nn import Module


logger = logging.get_logger(__name__)


class ALSTLossHandler:
    """Handles loss computation for ALST sequence parallel training.
    
    Following the pattern from ArcticTraining, this class handles:
    1. Detection of shift_labels vs labels in inputs
    2. Proper model forward pass for ALST
    3. Manual loss computation with sequence parallel aggregation
    """
    
    def __init__(self, sequence_parallel_group: Optional["dist.ProcessGroup"] = None):
        self.sp_group = sequence_parallel_group
    
    def compute_alst_loss(
        self, 
        model: "Module", 
        inputs: Dict[str, Any], 
        return_outputs: bool = False
    ) -> torch.Tensor:
        """Compute loss for ALST sequence parallel training.
        
        Args:
            model: The model to compute loss for
            inputs: Input batch (should contain shift_labels for ALST)
            return_outputs: Whether to return outputs (for compatibility)
            
        Returns:
            Loss tensor with proper sequence parallel aggregation
        """
        if "shift_labels" not in inputs:
            raise ValueError(
                "ALST mode requires 'shift_labels' in batch inputs. "
                "This should be created by UlyssesSPDataLoaderAdapter. "
                "Check that the ALST data adapter is properly applied."
            )
        
        if "labels" in inputs:
            logger.warning(
                "Found both 'labels' and 'shift_labels' in inputs. "
                "ALST mode uses 'shift_labels' - removing 'labels' for clarity."
            )
        
        # Extract shift_labels and prepare model inputs
        shift_labels = inputs["shift_labels"]
        model_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "shift_labels"]}
        
        # Forward pass through model
        outputs = model(**model_inputs, use_cache=False)
        
        # Extract logits from model outputs
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            available_attrs = dir(outputs) if hasattr(outputs, '__dict__') else "N/A"
            available_keys = list(outputs.keys()) if isinstance(outputs, dict) else "N/A"
            raise ValueError(
                f"Could not find logits in model outputs.\n"
                f"Output type: {type(outputs)}\n"
                f"Available attributes: {available_attrs}\n"
                f"Available keys: {available_keys}"
            )
        
        # Compute loss following ArcticTraining pattern
        loss = self._compute_shift_labels_loss(logits, shift_labels)
        
        # Aggregate loss across sequence parallel ranks
        if self.sp_group is not None and dist.is_initialized():
            loss = self._aggregate_loss_across_ranks(loss, shift_labels)
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def _compute_shift_labels_loss(self, logits: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with shift_labels.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            shift_labels: Pre-shifted labels [batch_size, seq_len]
            
        Returns:
            Loss tensor (scalar)
        """
        loss_fct = CrossEntropyLoss(reduction="sum", ignore_index=-100)
        
        # Flatten for cross-entropy computation
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # Compute loss
        loss_sum = loss_fct(flat_logits, flat_labels)
        
        # Normalize by number of valid tokens
        valid_tokens = (flat_labels != -100).sum()
        if valid_tokens > 0:
            loss = loss_sum / valid_tokens
        else:
            # No valid tokens in this shard - zero loss
            loss = loss_sum * 0.0
        
        return loss
    
    def _aggregate_loss_across_ranks(self, loss: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
        """Aggregate loss across sequence parallel ranks with proper weighting.
        
        Following ArcticTraining pattern for differentiable weighted averaging.
        
        Args:
            loss: Local rank loss
            shift_labels: Local rank shift_labels for counting valid tokens
            
        Returns:
            Properly aggregated loss across all sequence parallel ranks
        """
        # Count valid tokens on this rank
        valid_tokens = (shift_labels != -100).sum().float()
        
        # Gather losses and token counts from all ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=self.sp_group)
        valid_tokens_per_rank = torch.distributed.nn.functional.all_gather(valid_tokens, group=self.sp_group)
        
        # Compute weighted average
        # Each rank contributes proportionally to its number of valid tokens
        total_weighted_loss = sum(
            losses_per_rank[i] * valid_tokens_per_rank[i] 
            for i in range(len(losses_per_rank))
        )
        total_valid_tokens = sum(valid_tokens_per_rank)
        
        if total_valid_tokens > 0:
            aggregated_loss = total_weighted_loss / total_valid_tokens
        else:
            # No valid tokens across any rank - return zero loss
            aggregated_loss = total_weighted_loss * 0.0
        
        return aggregated_loss


def create_alst_loss_handler(sequence_parallel_group: Optional["dist.ProcessGroup"] = None) -> ALSTLossHandler:
    """Factory function to create ALST loss handler.
    
    Args:
        sequence_parallel_group: The sequence parallel process group
        
    Returns:
        Configured ALSTLossHandler instance
    """
    handler = ALSTLossHandler(sequence_parallel_group)
    logger.info_rank0("Created ALST loss handler for sequence parallel training")
    return handler


def should_use_alst_loss(inputs: Dict[str, Any], sequence_parallel_group: Optional["dist.ProcessGroup"]) -> bool:
    """Determine if ALST loss computation should be used.
    
    Args:
        inputs: Input batch
        sequence_parallel_group: Sequence parallel process group
        
    Returns:
        True if ALST loss should be used, False for standard loss computation
    """
    return (
        sequence_parallel_group is not None and
        "shift_labels" in inputs
    )