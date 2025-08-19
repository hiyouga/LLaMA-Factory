"""
FSDP Configuration Validation

Provides Pydantic-based validation for FSDP configuration parameters to catch
invalid or deprecated parameters before training starts.
"""

from enum import Enum
from typing import Any, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import yaml


class FsdpShardingStrategy(str, Enum):
    """Valid FSDP sharding strategies."""
    FULL_SHARD = "FULL_SHARD"
    SHARD_GRAD_OP = "SHARD_GRAD_OP" 
    NO_SHARD = "NO_SHARD"
    HYBRID_SHARD = "HYBRID_SHARD"
    _HYBRID_SHARD_ZERO2 = "_HYBRID_SHARD_ZERO2"


class FsdpBackwardPrefetchPolicy(str, Enum):
    """Valid FSDP backward prefetch policies."""
    BACKWARD_PRE = "BACKWARD_PRE"
    BACKWARD_POST = "BACKWARD_POST"
    NO_PREFETCH = "NO_PREFETCH"


class FsdpAutoWrapPolicy(str, Enum):
    """Valid FSDP auto wrap policies."""
    TRANSFORMER_BASED_WRAP = "TRANSFORMER_BASED_WRAP"
    SIZE_BASED_WRAP = "SIZE_BASED_WRAP"
    NO_WRAP = "NO_WRAP"


class FsdpStateDict(str, Enum):
    """Valid FSDP state dict types."""
    FULL_STATE_DICT = "FULL_STATE_DICT"
    LOCAL_STATE_DICT = "LOCAL_STATE_DICT"
    SHARDED_STATE_DICT = "SHARDED_STATE_DICT"


class DistributedType(str, Enum):
    """Valid distributed types."""
    FSDP = "FSDP"
    DEEPSPEED = "DEEPSPEED"
    MULTI_GPU = "MULTI_GPU"


class MixedPrecision(str, Enum):
    """Valid mixed precision modes."""
    no = "no"
    fp16 = "fp16"
    bf16 = "bf16"


class FsdpConfigV1(BaseModel):
    """FSDP v1 configuration validation schema."""
    
    model_config = ConfigDict(use_enum_values=True, extra='forbid')
    
    fsdp_auto_wrap_policy: Optional[FsdpAutoWrapPolicy] = None
    fsdp_backward_prefetch: Optional[FsdpBackwardPrefetchPolicy] = None
    fsdp_cpu_ram_efficient_loading: Optional[bool] = True
    fsdp_forward_prefetch: Optional[bool] = False
    fsdp_offload_params: Optional[bool] = False
    fsdp_reshard_after_forward: Optional[bool] = True
    fsdp_sharding_strategy: Optional[FsdpShardingStrategy] = FsdpShardingStrategy.FULL_SHARD
    fsdp_state_dict_type: Optional[FsdpStateDict] = FsdpStateDict.FULL_STATE_DICT
    fsdp_sync_module_states: Optional[bool] = True
    fsdp_use_orig_params: Optional[bool] = True


class FsdpConfigV2(BaseModel):
    """FSDP v2 configuration validation schema."""
    
    model_config = ConfigDict(use_enum_values=True, extra='forbid')
    
    fsdp_cpu_ram_efficient_loading: Optional[bool] = True
    fsdp_offload_params: Optional[bool] = False
    fsdp_reshard_after_forward: Optional[bool] = True
    fsdp_state_dict_type: Optional[FsdpStateDict] = FsdpStateDict.SHARDED_STATE_DICT
    fsdp_version: Literal[2] = 2
    
    @field_validator('fsdp_version')
    @classmethod
    def validate_version(cls, v):
        if v != 2:
            raise ValueError("fsdp_version must be 2 for FSDP v2 config")
        return v


class AccelerateConfig(BaseModel):
    """Complete Accelerate configuration validation schema."""
    
    model_config = ConfigDict(use_enum_values=True, extra='forbid')
    
    compute_environment: Optional[str] = "LOCAL_MACHINE"
    debug: Optional[bool] = False
    distributed_type: DistributedType
    downcast_bf16: Optional[str] = "no"
    fsdp_config: Optional[Union[FsdpConfigV1, FsdpConfigV2]] = None
    machine_rank: Optional[int] = 0
    main_training_function: Optional[str] = "main"
    mixed_precision: Optional[MixedPrecision] = MixedPrecision.no
    num_machines: Optional[int] = 1
    num_processes: Optional[int] = 1
    rdzv_backend: Optional[str] = "static"
    same_network: Optional[bool] = True
    tpu_env: Optional[list] = Field(default_factory=list)
    tpu_use_cluster: Optional[bool] = False
    tpu_use_sudo: Optional[bool] = False
    use_cpu: Optional[bool] = False
    
    @model_validator(mode='after')
    def validate_fsdp_requirements(self):
        """Validate FSDP-specific requirements after all fields are processed."""
        # Check if FSDP config is required but missing
        if self.distributed_type == 'FSDP' or self.distributed_type == DistributedType.FSDP:
            if self.fsdp_config is None:
                raise ValueError("fsdp_config is required when distributed_type is FSDP")
        
        return self

    @field_validator('fsdp_config', mode='before')
    @classmethod
    def validate_fsdp_config(cls, v, info):
        values = info.data if info else {}
        # Skip validation if not FSDP
        distributed_type = values.get('distributed_type')
        if distributed_type != 'FSDP' and distributed_type != DistributedType.FSDP:
            return v
        
        # Skip if fsdp_config is None - this will be caught by model validator
        if v is None:
            return v
            
        # Determine FSDP version and validate accordingly
        fsdp_version = v.get('fsdp_version', 1)
        
        if fsdp_version == 2:
            # Check for deprecated v1 parameters
            deprecated_params = [
                'fsdp_auto_wrap_policy', 'fsdp_backward_prefetch', 'fsdp_forward_prefetch',
                'fsdp_sync_module_states', 'fsdp_use_orig_params'
            ]
            found_deprecated = [param for param in deprecated_params if param in v]
            if found_deprecated:
                raise ValueError(
                    f"FSDP v2 does not support these parameters: {found_deprecated}. "
                    f"Remove them from your config."
                )
            return FsdpConfigV2(**v)
        else:
            return FsdpConfigV1(**v)


class FsdpValidator:
    """Utility class for FSDP configuration validation."""
    
    @staticmethod
    def validate_config_file(config_path: str) -> AccelerateConfig:
        """Validate an accelerate config file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            return AccelerateConfig(**config_dict)
            
        except FileNotFoundError:
            raise ValueError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    @staticmethod
    def validate_config_dict(config_dict: Dict[str, Any]) -> AccelerateConfig:
        """Validate an accelerate config dictionary."""
        return AccelerateConfig(**config_dict)
    
    @staticmethod
    def get_validation_errors(config_path: str) -> Optional[str]:
        """Get validation errors as a formatted string, if any."""
        try:
            FsdpValidator.validate_config_file(config_path)
            return None
        except Exception as e:
            return str(e)
    
    @staticmethod
    def suggest_fsdp_v2_migration(config_dict: Dict[str, Any]) -> Dict[str, str]:
        """Suggest migration steps for FSDP v1 to v2."""
        suggestions = {}
        
        fsdp_config = config_dict.get('fsdp_config', {})
        if fsdp_config.get('fsdp_version') != 2:
            suggestions['version'] = "Add 'fsdp_version: 2' to enable FSDP v2"
        
        deprecated_params = {
            'fsdp_auto_wrap_policy': "Remove - auto wrapping is handled automatically in v2",
            'fsdp_backward_prefetch': "Remove - backward prefetching is optimized automatically in v2", 
            'fsdp_forward_prefetch': "Remove - not yet implemented in v2",
            'fsdp_sync_module_states': "Remove - module state syncing is handled automatically in v2",
            'fsdp_use_orig_params': "Remove - DTensor handles original parameters automatically in v2"
        }
        
        for param, message in deprecated_params.items():
            if param in fsdp_config:
                suggestions[param] = message
        
        # Recommend SHARDED_STATE_DICT for v2
        if fsdp_config.get('fsdp_state_dict_type') != 'SHARDED_STATE_DICT':
            suggestions['state_dict'] = "Consider using 'SHARDED_STATE_DICT' for better v2 performance"
        
        return suggestions


def validate_fsdp_config_file(config_path: str) -> None:
    """
    Validate an FSDP config file and raise detailed errors if invalid.
    
    Args:
        config_path: Path to the accelerate config file
        
    Raises:
        ValueError: If the config is invalid with detailed error message
    """
    errors = FsdpValidator.get_validation_errors(config_path)
    if errors:
        raise ValueError(f"Invalid FSDP configuration in {config_path}:\n{errors}")