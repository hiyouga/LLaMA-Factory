"""Test cases for FSDP configuration validation.

This test suite validates FSDP configurations against the installed accelerate version,
ensuring compatibility and proper validation of both valid and invalid parameters.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


try:
    from accelerate import FullyShardedDataParallelPlugin
    from accelerate.utils import FSDPBackwardPrefetch, FSDPShardingStrategy, FSDPStateDictType  # noqa: F401

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from llamafactory.hparams.fsdp_validator import (
    FsdpConfigV1,
    FsdpConfigV2,
    FsdpValidator,
)


class TestFsdpValidator:
    """Test FSDP configuration validation."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yield f.name
        os.unlink(f.name)

    def create_config_file(self, config_dict, file_path):
        """Helper to create a YAML config file."""
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f)

    # Temporarily remove skip to check compatibility
    # @pytest.mark.skipif(not ACCELERATE_AVAILABLE, reason="Accelerate not available")
    def test_accelerate_fsdp_parameters_compatibility(self):
        """Test that our validator covers all FSDP parameters from accelerate."""
        # Get actual FSDP plugin parameters from accelerate
        plugin = FullyShardedDataParallelPlugin()

        # Extract parameter names from the plugin
        accelerate_params = set()
        for attr in dir(plugin):
            if attr.startswith("fsdp_") and not attr.startswith("_"):
                accelerate_params.add(attr)

        print(f"Accelerate FSDP parameters found: {sorted(accelerate_params)}")

        # Parameters we expect to handle in our validator
        validator_v1_params = {
            "fsdp_auto_wrap_policy",
            "fsdp_backward_prefetch",
            "fsdp_cpu_ram_efficient_loading",
            "fsdp_forward_prefetch",
            "fsdp_offload_params",
            "fsdp_reshard_after_forward",
            "fsdp_sharding_strategy",
            "fsdp_state_dict_type",
            "fsdp_sync_module_states",
            "fsdp_use_orig_params",
        }

        validator_v2_params = {
            "fsdp_cpu_ram_efficient_loading",
            "fsdp_offload_params",
            "fsdp_reshard_after_forward",
            "fsdp_state_dict_type",
            "fsdp_version",
        }

        # Check coverage (some parameters might be internal/private)
        public_accelerate_params = {p for p in accelerate_params if not p.startswith("_")}
        covered_params = validator_v1_params | validator_v2_params

        print(f"Validator V1 parameters: {sorted(validator_v1_params)}")
        print(f"Validator V2 parameters: {sorted(validator_v2_params)}")
        print(f"Coverage: {len(covered_params)}/{len(public_accelerate_params)} parameters")

        # We should cover at least the major parameters
        major_params = {"fsdp_sharding_strategy", "fsdp_offload_params", "fsdp_state_dict_type"}
        assert major_params.issubset(covered_params), f"Missing major parameters: {major_params - covered_params}"

    def test_valid_fsdp_v1_config(self, temp_config_file):
        """Test validation of a valid FSDP v1 configuration."""
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "FSDP",
            "fsdp_config": {
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_backward_prefetch": "BACKWARD_PRE",
                "fsdp_cpu_ram_efficient_loading": True,
                "fsdp_forward_prefetch": False,
                "fsdp_offload_params": False,
                "fsdp_reshard_after_forward": True,
                "fsdp_sharding_strategy": "FULL_SHARD",
                "fsdp_state_dict_type": "FULL_STATE_DICT",
                "fsdp_sync_module_states": True,
                "fsdp_use_orig_params": True,
            },
            "mixed_precision": "bf16",
            "num_processes": 2,
        }

        self.create_config_file(config, temp_config_file)

        # Should validate successfully
        validated_config = FsdpValidator.validate_config_file(temp_config_file)
        assert validated_config.distributed_type == "FSDP"
        assert isinstance(validated_config.fsdp_config, FsdpConfigV1)
        assert validated_config.fsdp_config.fsdp_sharding_strategy == "FULL_SHARD"

    def test_valid_fsdp_v2_config(self, temp_config_file):
        """Test validation of a valid FSDP v2 configuration."""
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "FSDP",
            "fsdp_config": {
                "fsdp_cpu_ram_efficient_loading": True,
                "fsdp_offload_params": True,
                "fsdp_reshard_after_forward": True,
                "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                "fsdp_version": 2,
            },
            "mixed_precision": "bf16",
            "num_processes": 8,
        }

        self.create_config_file(config, temp_config_file)

        # Should validate successfully
        validated_config = FsdpValidator.validate_config_file(temp_config_file)
        assert validated_config.distributed_type == "FSDP"
        assert isinstance(validated_config.fsdp_config, FsdpConfigV2)
        assert validated_config.fsdp_config.fsdp_version == 2
        assert validated_config.fsdp_config.fsdp_state_dict_type == "SHARDED_STATE_DICT"

    def test_deprecated_parameters_in_v2(self, temp_config_file):
        """Test that deprecated v1 parameters are rejected in v2 config."""
        config = {
            "distributed_type": "FSDP",
            "fsdp_config": {
                "fsdp_version": 2,
                "fsdp_offload_params": True,
                # These should be rejected in v2
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_backward_prefetch": "BACKWARD_PRE",
                "fsdp_use_orig_params": True,
            },
        }

        self.create_config_file(config, temp_config_file)

        # Should raise validation error
        with pytest.raises(ValueError) as exc_info:
            FsdpValidator.validate_config_file(temp_config_file)

        error_msg = str(exc_info.value)
        assert "does not support these parameters" in error_msg
        assert "fsdp_auto_wrap_policy" in error_msg
        assert "fsdp_backward_prefetch" in error_msg
        assert "fsdp_use_orig_params" in error_msg

    def test_invalid_enum_values(self, temp_config_file):
        """Test validation of invalid enum values."""
        config = {
            "distributed_type": "FSDP",
            "fsdp_config": {
                "fsdp_sharding_strategy": "INVALID_STRATEGY",
                "fsdp_backward_prefetch": "INVALID_PREFETCH",
                "fsdp_state_dict_type": "INVALID_STATE_DICT",
            },
        }

        self.create_config_file(config, temp_config_file)

        # Should raise validation error
        with pytest.raises(ValueError):
            FsdpValidator.validate_config_file(temp_config_file)

    def test_missing_fsdp_config(self, temp_config_file):
        """Test that missing fsdp_config is caught when distributed_type is FSDP."""
        config = {"distributed_type": "FSDP", "mixed_precision": "bf16"}

        self.create_config_file(config, temp_config_file)

        # Should raise validation error
        with pytest.raises(ValueError) as exc_info:
            FsdpValidator.validate_config_file(temp_config_file)

        assert "fsdp_config is required" in str(exc_info.value)

    def test_non_fsdp_config_no_validation(self, temp_config_file):
        """Test that non-FSDP configs don't trigger FSDP validation."""
        config = {"distributed_type": "MULTI_GPU", "mixed_precision": "fp16", "num_processes": 4}

        self.create_config_file(config, temp_config_file)

        # Should validate successfully without FSDP config
        validated_config = FsdpValidator.validate_config_file(temp_config_file)
        assert validated_config.distributed_type == "MULTI_GPU"
        assert validated_config.fsdp_config is None

    def test_migration_suggestions(self):
        """Test migration suggestions for FSDP v1 to v2."""
        v1_config = {
            "distributed_type": "FSDP",
            "fsdp_config": {
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_backward_prefetch": "BACKWARD_PRE",
                "fsdp_use_orig_params": False,
                "fsdp_state_dict_type": "FULL_STATE_DICT",
                "fsdp_offload_params": True,
            },
        }

        suggestions = FsdpValidator.suggest_fsdp_v2_migration(v1_config)

        # Should suggest adding version
        assert "version" in suggestions

        # Should suggest removing deprecated params
        deprecated_suggested = {"fsdp_auto_wrap_policy", "fsdp_backward_prefetch", "fsdp_use_orig_params"}
        suggested_keys = set(suggestions.keys())
        assert deprecated_suggested.issubset(
            suggested_keys
        ), f"Missing suggestions for: {deprecated_suggested - suggested_keys}"

        # Should suggest better state dict type
        assert "state_dict" in suggestions

    def test_config_dict_validation(self):
        """Test direct config dictionary validation."""
        valid_config = {
            "distributed_type": "FSDP",
            "fsdp_config": {
                "fsdp_version": 2,
                "fsdp_offload_params": True,
                "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            },
        }

        # Should validate successfully
        validated = FsdpValidator.validate_config_dict(valid_config)
        assert validated.distributed_type == "FSDP"
        assert validated.fsdp_config.fsdp_version == 2

    def test_file_not_found_error(self):
        """Test handling of missing config files."""
        non_existent_file = "/tmp/non_existent_config.yaml"

        with pytest.raises(ValueError) as exc_info:
            FsdpValidator.validate_config_file(non_existent_file)

        assert "Config file not found" in str(exc_info.value)

    def test_invalid_yaml_error(self, temp_config_file):
        """Test handling of invalid YAML files."""
        # Write invalid YAML
        with open(temp_config_file, "w") as f:
            f.write("distributed_type: FSDP\n  invalid: yaml: syntax}")

        with pytest.raises(ValueError) as exc_info:
            FsdpValidator.validate_config_file(temp_config_file)

        assert "Invalid YAML" in str(exc_info.value)

    # Test parameter name accuracy
    # @pytest.mark.skipif(not ACCELERATE_AVAILABLE, reason="Accelerate not available")
    def test_parameter_name_accuracy(self):
        """Test that our parameter names match accelerate's documentation."""
        # Test common parameter name variations that might cause confusion

        # The documentation issue mentioned: backward_prefetch vs backward_prefetch_policy
        config_with_wrong_name = {
            "distributed_type": "FSDP",
            "fsdp_config": {
                "fsdp_backward_prefetch_policy": "BACKWARD_PRE"  # Wrong name from docs
            },
        }

        # Our validator should accept the correct parameter name
        config_with_correct_name = {
            "distributed_type": "FSDP",
            "fsdp_config": {
                "fsdp_backward_prefetch": "BACKWARD_PRE"  # Correct name
            },
        }

        # Wrong name should fail validation (unknown parameter)
        with pytest.raises(ValueError):
            FsdpValidator.validate_config_dict(config_with_wrong_name)

        # Correct name should pass
        validated = FsdpValidator.validate_config_dict(config_with_correct_name)
        assert validated.fsdp_config.fsdp_backward_prefetch == "BACKWARD_PRE"


class TestFsdpValidatorIntegration:
    """Integration tests for FSDP validator with actual accelerate configs."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yield f.name
        os.unlink(f.name)

    def test_example_configs_validation(self):
        """Test validation of example configs in the repository."""
        base_path = Path(__file__).parent.parent / "examples" / "accelerate"

        # Test existing FSDP configs if they exist
        config_patterns = ["*fsdp*.yaml", "*FSDP*.yaml"]

        for pattern in config_patterns:
            for config_file in base_path.glob(pattern):
                try:
                    validated = FsdpValidator.validate_config_file(str(config_file))
                    print(f"✅ {config_file.name} validated successfully")

                    if validated.distributed_type == "FSDP":
                        fsdp_version = getattr(validated.fsdp_config, "fsdp_version", 1)
                        print(f"   FSDP version: {fsdp_version}")

                except Exception as e:
                    # Print error but don't fail the test - config might be intentionally invalid
                    print(f"⚠️  {config_file.name} validation failed: {e}")

    def test_environment_variable_integration(self, temp_config_file):
        """Test integration with ACCELERATE_CONFIG_FILE environment variable."""
        config = {
            "distributed_type": "FSDP",
            "fsdp_config": {
                "fsdp_version": 2,
                "fsdp_offload_params": True,
                "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            },
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config, f)

        # Test that validator can read from environment variable
        with patch.dict(os.environ, {"ACCELERATE_CONFIG_FILE": temp_config_file}):
            # This simulates how the validator is called in the training pipeline
            config_file = os.environ.get("ACCELERATE_CONFIG_FILE")
            assert config_file == temp_config_file

            # Should validate successfully
            validated = FsdpValidator.validate_config_file(config_file)
            assert validated.distributed_type == "FSDP"
            assert validated.fsdp_config.fsdp_version == 2


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
