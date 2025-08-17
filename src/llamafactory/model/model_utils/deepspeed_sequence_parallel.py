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

"""DeepSpeed Arctic Long Sequence Training (ALST) implementation for LLaMA-Factory."""

import importlib.util
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import PreTrainedModel

from ...extras import logging


if TYPE_CHECKING:
    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def check_alst_requirements() -> bool:
    """Check if ALST requirements are available."""
    try:
        import deepspeed
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
        
        # Check DeepSpeed version
        deepspeed_version = tuple(map(int, deepspeed.__version__.split('.')[:3]))
        if deepspeed_version < (0, 17, 4):
            logger.warning(f"DeepSpeed {deepspeed.__version__} found, but ALST requires 0.17.4+")
            return False
            
        return True
    except ImportError as e:
        logger.warning(f"ALST requirements not met: {e}")
        return False


def create_ulysses_sp_group(sp_size: int) -> Optional["dist.ProcessGroup"]:
    """Create Ulysses sequence parallel process group."""
    if not dist.is_initialized():
        logger.info_rank0("Distributed not initialized, cannot create SP group")
        return None
        
    world_size = dist.get_world_size()
    if world_size % sp_size != 0:
        raise ValueError(f"World size ({world_size}) must be divisible by sequence_parallel_size ({sp_size})")
    
    # Create sequence parallel groups
    sp_group_num = world_size // sp_size
    sp_ranks_list = [list(range(i * sp_size, (i + 1) * sp_size)) for i in range(sp_group_num)]
    
    sp_groups = []
    for sp_ranks in sp_ranks_list:
        group = dist.new_group(sp_ranks)
        sp_groups.append(group)
    
    # Find the group for current rank
    global_rank = dist.get_rank()
    sp_group_idx = global_rank // sp_size
    current_sp_group = sp_groups[sp_group_idx]
    
    logger.info_rank0(f"Created {sp_group_num} sequence parallel groups of size {sp_size}")
    logger.info_rank0(f"Current rank {global_rank} assigned to SP group {sp_group_idx}")
    
    return current_sp_group


class ALSTAttentionWrapper:
    """Wrapper for DeepSpeed ALST attention modules."""
    
    def __init__(
        self, 
        model_args: "ModelArguments",
        sp_group: "dist.ProcessGroup",
        model_config: Any
    ):
        self.model_args = model_args
        self.sp_group = sp_group
        self.model_config = model_config
        self.ulysses_attention = None
        
        # Initialize UlyssesSPAttentionHF if requirements are met
        if check_alst_requirements():
            self._initialize_ulysses_attention()
    
    def _initialize_ulysses_attention(self) -> None:
        """Initialize UlyssesSPAttentionHF for ALST."""
        try:
            from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
            
            # Calculate attention parameters
            hidden_size = getattr(self.model_config, 'hidden_size', 4096)
            num_attention_heads = getattr(self.model_config, 'num_attention_heads', 32)
            num_key_value_heads = getattr(self.model_config, 'num_key_value_heads', num_attention_heads)
            attn_head_size = hidden_size // num_attention_heads
            
            # Get sequence parameters  
            ulysses_degree = self.model_args.alst_ulysses_degree or self.model_args.sequence_parallel_size
            max_position_embeddings = getattr(self.model_config, 'max_position_embeddings', 32768)
            local_seq_length = max_position_embeddings // ulysses_degree
            
            logger.info_rank0(f"Initializing UlyssesSPAttentionHF with:")
            logger.info_rank0(f"  - Ulysses degree: {ulysses_degree}")
            logger.info_rank0(f"  - Global sequence length: {max_position_embeddings}")
            logger.info_rank0(f"  - Local sequence length: {local_seq_length}")
            logger.info_rank0(f"  - Attention heads: {num_attention_heads}")
            logger.info_rank0(f"  - KV heads: {num_key_value_heads}")
            
            self.ulysses_attention = UlyssesSPAttentionHF(
                attn=None,  # Will be set when wrapping actual attention
                local_seq_length=local_seq_length,
                global_seq_length=max_position_embeddings,
                batch_size=1,  # Will be dynamic during training
                attn_head_count=num_attention_heads,
                attn_head_size=attn_head_size,
                kv_head_count=num_key_value_heads,
                process_group=self.sp_group,
                seq_length_is_variable=True,
            )
            
            logger.info_rank0("UlyssesSPAttentionHF initialized successfully")
            
        except Exception as e:
            logger.info_rank0(f"Failed to initialize UlyssesSPAttentionHF: {e}")
            self.ulysses_attention = None
    
    def wrap_attention_module(self, attention_module: Any) -> Any:
        """Wrap an attention module with ALST capabilities."""
        if self.ulysses_attention is None:
            logger.warning("UlyssesSPAttentionHF not available, returning original module")
            return attention_module
            
        # Store original attention function
        original_forward = attention_module.forward
        
        def alst_attention_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs
        ) -> tuple[torch.Tensor, ...]:
            """ALST-enabled attention forward pass."""
            # Get batch size and sequence length
            batch_size, seq_len = hidden_states.shape[:2]
            
            # Update UlyssesSPAttentionHF configuration for current batch
            self.ulysses_attention.batch_size = batch_size
            if hasattr(self.ulysses_attention, 'local_seq_length'):
                # Update local sequence length based on actual input
                global_seq_len = seq_len * dist.get_world_size(self.sp_group)
                self.ulysses_attention.local_seq_length = seq_len
                self.ulysses_attention.global_seq_length = global_seq_len
            
            # Use UlyssesSPAttentionHF if sequence is long enough
            if seq_len >= 1024 and self.model_args.alst_sequence_backend == "deepspeed":
                try:
                    # Set the core attention function
                    self.ulysses_attention.attn = original_forward
                    
                    # Call ALST attention
                    return self.ulysses_attention(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **kwargs
                    )
                except Exception as e:
                    logger.warning(f"ALST attention failed, falling back to original: {e}")
                    return original_forward(
                        hidden_states, 
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **kwargs
                    )
            else:
                # Use original attention for shorter sequences or manual backend
                return original_forward(
                    hidden_states,
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    **kwargs
                )
        
        # Replace the forward method
        attention_module.forward = alst_attention_forward
        return attention_module


class DeepSpeedSequenceParallel:
    """Unified DeepSpeed ALST sequence parallelism manager."""
    
    def __init__(self, model_args: "ModelArguments"):
        self.model_args = model_args
        self.sp_group: Optional["dist.ProcessGroup"] = None
        self.attention_wrapper: Optional[ALSTAttentionWrapper] = None
        self.is_initialized = False
        
    def should_use_alst(self) -> bool:
        """Check if ALST should be used based on configuration."""
        return (
            self.model_args.sequence_parallel_size > 1 and
            self.model_args.sequence_parallel_mode == "deepspeed-alst" and
            self.model_args.alst_sequence_backend == "deepspeed" and
            check_alst_requirements()
        )
    
    def initialize_sp_group(self) -> Optional["dist.ProcessGroup"]:
        """Initialize sequence parallel process group."""
        if not self.should_use_alst():
            logger.info_rank0("ALST not enabled or requirements not met")
            return None
            
        if self.sp_group is not None:
            logger.info_rank0("Sequence parallel group already initialized")
            return self.sp_group
            
        try:
            self.sp_group = create_ulysses_sp_group(self.model_args.sequence_parallel_size)
            self.is_initialized = True
            logger.info_rank0("DeepSpeed ALST sequence parallel group initialized")
            return self.sp_group
        except Exception as e:
            logger.info_rank0(f"Failed to initialize ALST SP group: {e}")
            return None
    
    def wrap_model_attention(self, model: PreTrainedModel) -> None:
        """Wrap model attention modules with ALST capabilities."""
        if not self.is_initialized or self.sp_group is None:
            logger.warning("ALST not initialized, skipping attention wrapping")
            return
            
        try:
            # Create attention wrapper
            self.attention_wrapper = ALSTAttentionWrapper(
                self.model_args, 
                self.sp_group, 
                model.config
            )
            
            # Find and wrap attention modules
            wrapped_count = 0
            for name, module in model.named_modules():
                # Look for common attention module patterns
                if any(pattern in name.lower() for pattern in ['attention', 'attn', 'self_attn']):
                    if hasattr(module, 'forward'):
                        logger.info_rank0(f"Wrapping attention module: {name}")
                        self.attention_wrapper.wrap_attention_module(module)
                        wrapped_count += 1
                        
            logger.info_rank0(f"Successfully wrapped {wrapped_count} attention modules with ALST")
            
        except Exception as e:
            logger.info_rank0(f"Failed to wrap model attention with ALST: {e}")
    
    def get_data_parallel_config(self) -> dict[str, Any]:
        """Get data parallel configuration for ALST."""
        if not self.is_initialized:
            return {}
            
        return {
            "sequence_parallel_size": self.model_args.sequence_parallel_size,
            "sequence_parallel_mode": "deepspeed-alst",
            "ulysses_degree": self.model_args.alst_ulysses_degree or self.model_args.sequence_parallel_size,
            "sequence_tiling_enabled": self.model_args.alst_sequence_tiling,
            "memory_optimizations_enabled": self.model_args.alst_memory_optimizations,
        }


def apply_deepspeed_sequence_parallel(model_args: "ModelArguments", model: PreTrainedModel) -> Optional["dist.ProcessGroup"]:
    """Apply DeepSpeed ALST sequence parallelism to model."""
    if model_args.sequence_parallel_size <= 1:
        return None
        
    # Create DeepSpeed SP manager
    ds_sp = DeepSpeedSequenceParallel(model_args)
    
    # Initialize SP group
    sp_group = ds_sp.initialize_sp_group()
    if sp_group is None:
        logger.info_rank0("DeepSpeed ALST initialization failed, falling back to manual SP")
        return None
        
    # Wrap model attention modules
    ds_sp.wrap_model_attention(model)
    
    # Store SP manager on model for later use
    model.deepspeed_sp_manager = ds_sp
    
    logger.info_rank0("DeepSpeed ALST sequence parallelism applied successfully")
    return sp_group