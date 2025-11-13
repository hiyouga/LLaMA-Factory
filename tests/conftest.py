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

from llamafactory.extras.misc import get_current_device, is_env_enabled
from llamafactory.train.test_utils import patch_valuehead_model


try:
    CURRENT_DEVICE = get_current_device().type
except Exception:
    CURRENT_DEVICE = "cpu"


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"' or set RUN_SLOW=1 to run)"
    )
    config.addinivalue_line(
        "markers", "skip_on_devices: skip test on specified devices, e.g., @pytest.mark.skip_on_devices('npu', 'xpu')"
    )
    config.addinivalue_line(
        "markers", "require_device: test requires specific device, e.g., @pytest.mark.require_device('cuda')"
    )


def _handle_slow_tests(items):
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


def _handle_device_skips(items):
    """Skip tests on specified devices based on skip_on_devices marker.

    Usage:
        @pytest.mark.skip_on_devices("npu", "xpu")
        def test_something():
            pass
    """
    for item in items:
        skip_marker = item.get_closest_marker("skip_on_devices")
        if skip_marker:
            skip_devices = skip_marker.args
            if CURRENT_DEVICE in skip_devices:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"test skipped on {CURRENT_DEVICE.upper()} (skip list: {', '.join(skip_devices)})"
                    )
                )


def _handle_device_requirements(items):
    """Skip tests that require a specific device when running on other devices.

    Usage:
        @pytest.mark.require_device("cuda")
        def test_gpu_only():
            pass
    """
    for item in items:
        require_marker = item.get_closest_marker("require_device")
        if require_marker:
            required_device = require_marker.args[0] if require_marker.args else None
            if required_device and CURRENT_DEVICE != required_device:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"test requires {required_device.upper()} (current: {CURRENT_DEVICE.upper()})"
                    )
                )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and environment."""
    _handle_slow_tests(items)
    _handle_device_skips(items)
    _handle_device_requirements(items)


@pytest.fixture
def fix_valuehead_cpu_loading():
    """Fix valuehead model loading."""
    patch_valuehead_model()
