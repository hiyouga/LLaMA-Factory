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

"""Configuration management for Arctic Long Sequence Training (ALST)."""

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...extras import logging


if TYPE_CHECKING:
    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


@dataclass
class ALSTConfig:
    """Configuration for Arctic Long Sequence Training."""
    
    # Core ALST parameters
    enabled: bool = False
    sequence_parallel_size: int = 1
    sequence_parallel_mode: str = "deepspeed-alst"
    sequence_backend: str = "deepspeed"
    
    # Ulysses parameters
    ulysses_degree: Optional[int] = None
    ulysses_seq_length_is_variable: bool = True
    
    # Sequence tiling parameters
    sequence_tiling: bool = False
    tiling_chunk_size: Optional[int] = None
    
    # Memory optimization parameters
    memory_optimizations: bool = True
    pytorch_profiling: bool = False
    
    # Flash Attention parameters
    flash_attention_version: str = "auto"  # "auto", "fa2", "fa3"
    enable_flash_attention_3: bool = False
    
    # DeepSpeed integration
    deepspeed_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Set ulysses_degree default
        if self.ulysses_degree is None:
            self.ulysses_degree = self.sequence_parallel_size
            
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate ALST configuration parameters."""
        if self.enabled and self.sequence_parallel_size <= 1:
            logger.warning("ALST enabled but sequence_parallel_size <= 1, disabling ALST")
            self.enabled = False
            
        if self.ulysses_degree and self.ulysses_degree > self.sequence_parallel_size:
            logger.warning(f"ulysses_degree ({self.ulysses_degree}) > sequence_parallel_size ({self.sequence_parallel_size}), adjusting")
            self.ulysses_degree = self.sequence_parallel_size
            
        if self.sequence_tiling and self.tiling_chunk_size is None:
            # Set reasonable default chunk size
            self.tiling_chunk_size = 8192
            logger.info_rank0(f"Set default tiling_chunk_size to {self.tiling_chunk_size}")
    
    def to_deepspeed_config(self) -> Dict[str, Any]:
        """Convert ALST config to DeepSpeed configuration format."""
        if not self.enabled:
            return {}
            
        config = {
            "sequence_parallel": {
                "enabled": True,
                "size": self.sequence_parallel_size,
                "mode": "ulysses",
                "ulysses": {
                    "degree": self.ulysses_degree,
                    "seq_length_is_variable": self.ulysses_seq_length_is_variable,
                }
            }
        }
        
        if self.sequence_tiling:
            config["sequence_parallel"]["tiling"] = {
                "enabled": True,
                "chunk_size": self.tiling_chunk_size,
            }
            
        if self.memory_optimizations:
            config["sequence_parallel"]["memory_optimizations"] = {
                "enabled": True,
                "pytorch_profiling": self.pytorch_profiling,
            }
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_model_args(cls, model_args: "ModelArguments") -> "ALSTConfig":
        """Create ALSTConfig from ModelArguments."""
        return cls(
            enabled=(
                model_args.sequence_parallel_size > 1 and
                model_args.sequence_parallel_mode == "deepspeed-alst" and
                model_args.alst_sequence_backend == "deepspeed"
            ),
            sequence_parallel_size=model_args.sequence_parallel_size,
            sequence_parallel_mode=model_args.sequence_parallel_mode,
            sequence_backend=model_args.alst_sequence_backend,
            ulysses_degree=model_args.alst_ulysses_degree,
            sequence_tiling=model_args.alst_sequence_tiling,
            memory_optimizations=model_args.alst_memory_optimizations,
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ALSTConfig":
        """Create ALSTConfig from dictionary."""
        return cls(**config_dict)
    
    @classmethod 
    def from_json(cls, json_str: str) -> "ALSTConfig":
        """Create ALSTConfig from JSON string."""
        return cls.from_dict(json.loads(json_str))


def create_alst_config(model_args: "ModelArguments") -> ALSTConfig:
    """Create and validate ALST configuration from model arguments."""
    config = ALSTConfig.from_model_args(model_args)
    
    logger.info_rank0(f"Created ALST config: enabled={config.enabled}")
    if config.enabled:
        logger.info_rank0(f"ALST configuration:")
        logger.info_rank0(f"  - Sequence parallel size: {config.sequence_parallel_size}")
        logger.info_rank0(f"  - Ulysses degree: {config.ulysses_degree}")
        logger.info_rank0(f"  - Sequence tiling: {config.sequence_tiling}")
        logger.info_rank0(f"  - Memory optimizations: {config.memory_optimizations}")
        
    return config


def validate_alst_requirements(config: ALSTConfig) -> bool:
    """Validate that ALST requirements are met."""
    if not config.enabled:
        return True
        
    try:
        import deepspeed
        
        # DeepSpeed version check removed - ALST works with 0.17.2+
            
        # Check for required modules - only UlyssesSPAttentionHF is needed per DeepSpeed docs
        try:
            from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
        except ImportError as e:
            logger.info_rank0(f"Required DeepSpeed ALST modules not available: {e}")
            return False
            
        # Check Flash Attention
        try:
            import flash_attn
            flash_version = getattr(flash_attn, '__version__', '0.0.0')
            if tuple(map(int, flash_version.split('.')[:2])) < (2, 0):
                logger.warning(f"Flash Attention {flash_version} found, recommend 2.0+ for optimal ALST performance")
        except ImportError:
            logger.warning("Flash Attention not found, may impact ALST performance")
            
        return True
        
    except ImportError:
        logger.info_rank0("DeepSpeed not available, cannot use ALST")
        return False


def update_deepspeed_config_for_alst(
    deepspeed_config: Dict[str, Any], 
    alst_config: ALSTConfig
) -> Dict[str, Any]:
    """Update DeepSpeed configuration with ALST settings."""
    if not alst_config.enabled:
        return deepspeed_config
        
    # Merge ALST config into DeepSpeed config
    alst_ds_config = alst_config.to_deepspeed_config()
    
    # Deep merge configurations
    updated_config = deepspeed_config.copy()
    for key, value in alst_ds_config.items():
        if key in updated_config and isinstance(updated_config[key], dict) and isinstance(value, dict):
            updated_config[key].update(value)
        else:
            updated_config[key] = value
            
    logger.info_rank0("Updated DeepSpeed configuration with ALST settings")
    return updated_config


def get_alst_environment_variables(config: ALSTConfig) -> Dict[str, str]:
    """Get environment variables needed for ALST."""
    env_vars = {}
    
    if not config.enabled:
        return env_vars
        
    # DeepSpeed ALST environment variables
    env_vars["DEEPSPEED_SEQUENCE_PARALLEL"] = "true"
    env_vars["DEEPSPEED_ULYSSES_DEGREE"] = str(config.ulysses_degree)
    
    if config.sequence_tiling:
        env_vars["DEEPSPEED_SEQUENCE_TILING"] = "true"
        if config.tiling_chunk_size:
            env_vars["DEEPSPEED_TILING_CHUNK_SIZE"] = str(config.tiling_chunk_size)
            
    if config.memory_optimizations:
        env_vars["DEEPSPEED_MEMORY_OPTIMIZATIONS"] = "true"
        
    if config.pytorch_profiling:
        env_vars["DEEPSPEED_PYTORCH_PROFILING"] = "true"
        
    return env_vars