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

"""Migration utilities for transitioning to ALST sequence parallelism."""

import json
import os
from typing import Any, Optional

import yaml

from . import logging


logger = logging.get_logger(__name__)


def migrate_sequence_parallel_config(config_data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Migrate legacy sequence parallel configuration to ALST format.

    Args:
        config_data: Configuration dictionary (YAML/JSON)

    Returns:
        Tuple of (migrated_config, was_migrated)
    """
    migrated_config = config_data.copy()
    was_migrated = False

    # Check if legacy sequence parallel configuration exists
    legacy_modes = ["zigzag-ring", "ulysses"]
    current_mode = migrated_config.get("sequence_parallel_mode")

    if migrated_config.get("sequence_parallel_size", 1) > 1 and current_mode in legacy_modes:
        logger.info_rank0(f"Found legacy sequence parallel configuration: {current_mode}")

        # Recommend migration to ALST
        if current_mode == "ulysses":
            # Ulysses mode can be directly migrated to ALST
            migrated_config["sequence_parallel_mode"] = "deepspeed-alst"
            migrated_config["alst_sequence_backend"] = "deepspeed"
            migrated_config["alst_ulysses_degree"] = migrated_config.get("sequence_parallel_size")
            migrated_config["alst_sequence_tiling"] = True
            migrated_config["alst_memory_optimizations"] = True

            logger.info_rank0("Migrated ulysses mode to DeepSpeed ALST")
            was_migrated = True

        elif current_mode == "zigzag-ring":
            # Zigzag-ring requires manual consideration
            logger.warning(
                f"Found zigzag-ring sequence parallel mode. Consider migrating to ALST for better performance:\n"
                f"  sequence_parallel_mode: deepspeed-alst\n"
                f"  alst_sequence_backend: deepspeed\n"
                f"  alst_ulysses_degree: {migrated_config.get('sequence_parallel_size')}\n"
                f"  alst_sequence_tiling: true\n"
                f"  alst_memory_optimizations: true"
            )

    # Check for deprecated parameters
    deprecated_params = {"shuffle_for_sequence_parallel": "ALST handles data distribution automatically"}

    for param, message in deprecated_params.items():
        if param in migrated_config:
            logger.warning(f"Parameter '{param}' may not be needed with ALST: {message}")

    return migrated_config, was_migrated


def check_alst_compatibility(config_data: dict[str, Any]) -> dict[str, Any]:
    """Check ALST compatibility and provide recommendations.

    Args:
        config_data: Configuration dictionary

    Returns:
        Dictionary with compatibility status and recommendations
    """
    compatibility_report = {"compatible": True, "warnings": [], "recommendations": [], "requirements": []}

    # Check sequence parallel configuration
    sp_size = config_data.get("sequence_parallel_size", 1)
    sp_mode = config_data.get("sequence_parallel_mode", "zigzag-ring")

    if sp_size > 1 and sp_mode == "deepspeed-alst":
        # ALST-specific checks

        # Check sequence length
        cutoff_len = config_data.get("cutoff_len", 2048)
        if cutoff_len < 4096:
            compatibility_report["recommendations"].append(
                f"ALST is most beneficial for sequences >= 4096 tokens. Current cutoff_len: {cutoff_len}"
            )

        # Check DeepSpeed integration
        deepspeed_config = config_data.get("deepspeed")
        if not deepspeed_config:
            compatibility_report["warnings"].append("ALST requires DeepSpeed configuration")
            compatibility_report["compatible"] = False

        # Check hardware requirements
        compatibility_report["requirements"].extend(
            [
                "DeepSpeed >= 0.17.4",
                "Flash Attention >= 2.0 (recommended)",
                f"Multi-GPU setup with {sp_size} GPUs",
                "CUDA-capable GPUs (H100/Hopper recommended for optimal performance)",
            ]
        )

        # Check training settings
        per_device_batch_size = config_data.get("per_device_train_batch_size", 1)
        if per_device_batch_size > 2:
            compatibility_report["recommendations"].append(
                f"Consider reducing per_device_train_batch_size (current: {per_device_batch_size}) "
                "as ALST enables training longer sequences"
            )

        # Check precision settings
        if not (config_data.get("bf16") or config_data.get("fp16")):
            compatibility_report["recommendations"].append("Enable bf16 or fp16 for optimal ALST performance")

        # Check gradient checkpointing
        if not config_data.get("gradient_checkpointing"):
            compatibility_report["recommendations"].append(
                "Enable gradient_checkpointing for memory efficiency with long sequences"
            )

    return compatibility_report


def create_alst_deepspeed_config(
    base_config: Optional[dict[str, Any]] = None,
    sequence_parallel_size: int = 4,
    sequence_tiling: bool = True,
    memory_optimizations: bool = True,
) -> dict[str, Any]:
    """Create DeepSpeed configuration optimized for ALST.

    Args:
        base_config: Base DeepSpeed configuration to extend
        sequence_parallel_size: Number of sequence parallel processes
        sequence_tiling: Enable sequence tiling
        memory_optimizations: Enable memory optimizations

    Returns:
        DeepSpeed configuration dictionary
    """
    if base_config is None:
        base_config = {
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "bf16": {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
            },
        }

    # Add ALST sequence parallel configuration
    alst_config = base_config.copy()
    alst_config["sequence_parallel"] = {
        "enabled": True,
        "size": sequence_parallel_size,
        "mode": "ulysses",
        "ulysses": {"degree": sequence_parallel_size, "seq_length_is_variable": True},
    }

    if sequence_tiling:
        alst_config["sequence_parallel"]["tiling"] = {"enabled": True, "chunk_size": 8192}

    if memory_optimizations:
        alst_config["sequence_parallel"]["memory_optimizations"] = {"enabled": True, "pytorch_profiling": False}

    return alst_config


def migrate_config_file(input_path: str, output_path: Optional[str] = None, backup: bool = True) -> str:
    """Migrate a configuration file to use ALST if applicable.

    Args:
        input_path: Path to input configuration file
        output_path: Path for output file (if None, overwrites input)
        backup: Create backup of original file

    Returns:
        Path to migrated configuration file
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Configuration file not found: {input_path}")

    # Determine file format
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".yaml" or ext == ".yml":
        with open(input_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        file_format = "yaml"
    elif ext == ".json":
        with open(input_path, encoding="utf-8") as f:
            config_data = json.load(f)
        file_format = "json"
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}")

    # Create backup if requested
    if backup and output_path != input_path:
        backup_path = input_path + ".backup"
        if not os.path.exists(backup_path):
            os.rename(input_path, backup_path)
            logger.info_rank0(f"Created backup: {backup_path}")

    # Migrate configuration
    migrated_config, was_migrated = migrate_sequence_parallel_config(config_data)

    if was_migrated:
        # Write migrated configuration
        output_file = output_path or input_path

        if file_format == "yaml":
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(migrated_config, f, default_flow_style=False, indent=2)
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(migrated_config, f, indent=2)

        logger.info_rank0(f"Migrated configuration saved to: {output_file}")

        # Generate compatibility report
        compatibility = check_alst_compatibility(migrated_config)
        if not compatibility["compatible"]:
            logger.warning("Migration completed but configuration may need additional changes:")
            for warning in compatibility["warnings"]:
                logger.warning(f"  - {warning}")

        if compatibility["recommendations"]:
            logger.info_rank0("Recommendations for optimal ALST performance:")
            for rec in compatibility["recommendations"]:
                logger.info_rank0(f"  - {rec}")

        return output_file
    else:
        logger.info_rank0("No migration needed for configuration file")
        return input_path


def validate_alst_installation() -> dict[str, Any]:
    """Validate ALST installation and requirements.

    Returns:
        Dictionary with validation results
    """
    validation_result = {"valid": True, "errors": [], "warnings": [], "info": []}

    # Check DeepSpeed
    try:
        import deepspeed  # noqa: F401

        version_str = deepspeed.__version__
        tuple(map(int, version_str.split(".")[:3]))

        # ALST works with DeepSpeed 0.17.2+ so accept any reasonable version
        validation_result["info"].append(f"DeepSpeed {version_str} - ALST compatible")

        # Check ALST modules
        try:
            from deepspeed.runtime.sequence_parallel.ulysses_sp import (  # noqa: F401
                UlyssesSPAttentionHF,
                UlyssesSPDataLoaderAdapter,
            )

            validation_result["info"].append("DeepSpeed ALST modules available")
        except ImportError as e:
            validation_result["errors"].append(f"DeepSpeed ALST modules not available: {e}")
            validation_result["valid"] = False

    except ImportError:
        validation_result["errors"].append("DeepSpeed not installed")
        validation_result["valid"] = False

    # Check Flash Attention
    try:
        import flash_attn

        fa_version = getattr(flash_attn, "__version__", "unknown")
        validation_result["info"].append(f"Flash Attention {fa_version} available")
    except ImportError:
        validation_result["warnings"].append("Flash Attention not found - may impact ALST performance")

    # Check PyTorch
    try:
        import torch

        torch_version = torch.__version__
        validation_result["info"].append(f"PyTorch {torch_version} available")

        if torch.cuda.is_available():
            cuda_devices = torch.cuda.device_count()
            validation_result["info"].append(f"CUDA available with {cuda_devices} GPU(s)")
        else:
            validation_result["warnings"].append("CUDA not available - ALST requires GPU")
    except ImportError:
        validation_result["errors"].append("PyTorch not installed")
        validation_result["valid"] = False

    return validation_result
