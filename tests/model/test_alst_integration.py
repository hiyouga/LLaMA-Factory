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

"""Tests for Arctic Long Sequence Training (ALST) integration."""

from unittest.mock import Mock, patch

import pytest

from llamafactory.extras.migration_utils import migrate_sequence_parallel_config, validate_alst_installation
from llamafactory.hparams import ModelArguments
from llamafactory.model.model_utils.alst_config import ALSTConfig, create_alst_config, validate_alst_requirements
from llamafactory.model.model_utils.deepspeed_sequence_parallel import (
    DeepSpeedSequenceParallel,
    check_alst_requirements,
)


class TestALSTConfig:
    """Test ALST configuration management."""

    def test_alst_config_creation(self):
        """Test ALST config creation from ModelArguments."""
        model_args = Mock(spec=ModelArguments)
        model_args.sequence_parallel_size = 4
        model_args.sequence_parallel_mode = "deepspeed-alst"
        model_args.alst_sequence_backend = "deepspeed"
        model_args.alst_ulysses_degree = None
        model_args.alst_sequence_tiling = True
        model_args.alst_memory_optimizations = True

        config = create_alst_config(model_args)

        assert config.enabled
        assert config.sequence_parallel_size == 4
        assert config.ulysses_degree == 4  # Should default to sequence_parallel_size
        assert config.sequence_tiling
        assert config.memory_optimizations

    def test_alst_config_disabled(self):
        """Test ALST config when disabled."""
        model_args = Mock(spec=ModelArguments)
        model_args.sequence_parallel_size = 1
        model_args.sequence_parallel_mode = "zigzag-ring"
        model_args.alst_sequence_backend = "manual"
        model_args.alst_ulysses_degree = None
        model_args.alst_sequence_tiling = False
        model_args.alst_memory_optimizations = True

        config = create_alst_config(model_args)

        assert not config.enabled

    def test_alst_config_validation(self):
        """Test ALST configuration validation."""
        config = ALSTConfig(
            enabled=True,
            sequence_parallel_size=4,
            ulysses_degree=8,  # Should be adjusted to match SP size
        )

        # Validation should adjust ulysses_degree
        assert config.ulysses_degree == 4

    def test_deepspeed_config_generation(self):
        """Test DeepSpeed config generation from ALST config."""
        config = ALSTConfig(enabled=True, sequence_parallel_size=4, sequence_tiling=True, memory_optimizations=True)

        ds_config = config.to_deepspeed_config()

        assert "sequence_parallel" in ds_config
        assert ds_config["sequence_parallel"]["enabled"]
        assert ds_config["sequence_parallel"]["size"] == 4
        assert "tiling" in ds_config["sequence_parallel"]
        assert "memory_optimizations" in ds_config["sequence_parallel"]


class TestDeepSpeedSequenceParallel:
    """Test DeepSpeed sequence parallel management."""

    @patch("llamafactory.model.model_utils.deepspeed_sequence_parallel.check_alst_requirements")
    def test_should_use_alst(self, mock_check_requirements):
        """Test ALST usage decision logic."""
        mock_check_requirements.return_value = True

        model_args = Mock(spec=ModelArguments)
        model_args.sequence_parallel_size = 4
        model_args.sequence_parallel_mode = "deepspeed-alst"
        model_args.alst_sequence_backend = "deepspeed"

        ds_sp = DeepSpeedSequenceParallel(model_args)

        assert ds_sp.should_use_alst()

    @patch("llamafactory.model.model_utils.deepspeed_sequence_parallel.check_alst_requirements")
    def test_should_not_use_alst_legacy_mode(self, mock_check_requirements):
        """Test ALST not used for legacy modes."""
        mock_check_requirements.return_value = True

        model_args = Mock(spec=ModelArguments)
        model_args.sequence_parallel_size = 4
        model_args.sequence_parallel_mode = "zigzag-ring"  # Legacy mode
        model_args.alst_sequence_backend = "deepspeed"

        ds_sp = DeepSpeedSequenceParallel(model_args)

        assert not ds_sp.should_use_alst()

    @patch("llamafactory.model.model_utils.deepspeed_sequence_parallel.check_alst_requirements")
    def test_should_not_use_alst_no_requirements(self, mock_check_requirements):
        """Test ALST not used when requirements not met."""
        mock_check_requirements.return_value = False

        model_args = Mock(spec=ModelArguments)
        model_args.sequence_parallel_size = 4
        model_args.sequence_parallel_mode = "deepspeed-alst"
        model_args.alst_sequence_backend = "deepspeed"

        ds_sp = DeepSpeedSequenceParallel(model_args)

        assert not ds_sp.should_use_alst()


class TestMigrationUtils:
    """Test configuration migration utilities."""

    def test_migrate_ulysses_to_alst(self):
        """Test migration from ulysses mode to ALST."""
        config_data = {"sequence_parallel_size": 4, "sequence_parallel_mode": "ulysses", "cutoff_len": 32768}

        migrated_config, was_migrated = migrate_sequence_parallel_config(config_data)

        assert was_migrated
        assert migrated_config["sequence_parallel_mode"] == "deepspeed-alst"
        assert migrated_config["alst_sequence_backend"] == "deepspeed"
        assert migrated_config["alst_ulysses_degree"] == 4
        assert migrated_config["alst_sequence_tiling"]

    def test_no_migration_needed(self):
        """Test no migration when not applicable."""
        config_data = {"sequence_parallel_size": 1, "cutoff_len": 2048}

        migrated_config, was_migrated = migrate_sequence_parallel_config(config_data)

        assert not was_migrated
        assert migrated_config == config_data

    def test_zigzag_ring_warning(self):
        """Test warning for zigzag-ring mode."""
        config_data = {"sequence_parallel_size": 4, "sequence_parallel_mode": "zigzag-ring", "cutoff_len": 16384}

        migrated_config, was_migrated = migrate_sequence_parallel_config(config_data)

        # Should not auto-migrate zigzag-ring, but should issue warning
        assert not was_migrated
        assert migrated_config["sequence_parallel_mode"] == "zigzag-ring"


@pytest.mark.skipif(not check_alst_requirements(), reason="ALST requirements not available")
class TestALSTIntegration:
    """Integration tests for ALST (requires DeepSpeed)."""

    def test_alst_requirements_check(self):
        """Test ALST requirements validation."""
        result = check_alst_requirements()
        assert isinstance(result, bool)

    def test_validate_alst_requirements(self):
        """Test ALST requirements validation from config."""
        config = ALSTConfig(enabled=True, sequence_parallel_size=2)
        result = validate_alst_requirements(config)
        assert isinstance(result, bool)

    def test_installation_validation(self):
        """Test ALST installation validation."""
        result = validate_alst_installation()

        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "info" in result
        assert isinstance(result["valid"], bool)
