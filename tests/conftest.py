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

"""LLaMA-Factory test configuration.

Contains shared fixtures, pytest configuration, and custom markers.
"""

import pytest
from pytest import Config, Item

from llamafactory.extras.misc import get_current_device, is_env_enabled
from llamafactory.train.test_utils import patch_valuehead_model


try:
    CURRENT_DEVICE = get_current_device().type
except Exception:
    CURRENT_DEVICE = "cpu"


def pytest_configure(config: Config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"' or set RUN_SLOW=1 to run)"
    )
    config.addinivalue_line("markers", "runs_on: test requires specific device, e.g., @pytest.mark.runs_on(['cpu'])")


def _handle_runs_on(items: list[Item]):
    """Skip tests on specified devices based on runs_on marker.

    Usage:
        # Skip tests on specified devices
        @pytest.mark.runs_on(['cpu'])
        def test_something():
            pass
    """
    for item in items:
        runs_on_marker = item.get_closest_marker("runs_on")
        if runs_on_marker:
            runs_on_devices = runs_on_marker.args[0]

            # Compatibility handling: Allow a single string instead of a list
            # Example: @pytest.mark.("cpu")
            if isinstance(runs_on_devices, str):
                runs_on_devices = [runs_on_devices]

            if CURRENT_DEVICE not in runs_on_devices:
                item.add_marker(
                    pytest.mark.skip(reason=f"test requires one of {runs_on_devices} (current: {CURRENT_DEVICE})")
                )


def _handle_slow_tests(items: list[Item]):
    """Skip slow tests unless RUN_SLOW environment variable is set.

    Usage:
        # Skip slow tests (default)
        @pytest.mark.slow

        # Run slow tests
        RUN_SLOW=1 pytest tests/
    """
    if not is_env_enabled("RUN_SLOW", "0"):
        skip_slow = pytest.mark.skip(reason="slow test (set RUN_SLOW=1 to run)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_collection_modifyitems(config: Config, items: list[Item]):
    """Modify test collection based on markers and environment."""
    _handle_slow_tests(items)
    _handle_runs_on(items)


@pytest.fixture
def fix_valuehead_cpu_loading():
    """Fix valuehead model loading."""
    patch_valuehead_model()
