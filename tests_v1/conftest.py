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

import os

import pytest
from pytest import Config, Item

from llamafactory.extras.misc import get_current_device, is_env_enabled
from llamafactory.train.test_utils import patch_valuehead_model


try:
    CURRENT_DEVICE = get_current_device().type  # cpu | cuda | npu
except Exception:
    CURRENT_DEVICE = "cpu"


def pytest_configure(config: Config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"' or set RUN_SLOW=1 to run)",
    )
    config.addinivalue_line(
        "markers",
        "runs_on: test requires specific device type, e.g., @pytest.mark.runs_on(['cuda'])",
    )
    config.addinivalue_line(
        "markers",
        "require_distributed(num_devices): allow multi-device execution (default: 2)",
    )


def _handle_runs_on(items: list[Item]):
    """Skip tests on specified device TYPES (cpu/cuda/npu)."""
    for item in items:
        marker = item.get_closest_marker("runs_on")
        if not marker:
            continue

        devices = marker.args[0]
        if isinstance(devices, str):
            devices = [devices]

        if CURRENT_DEVICE not in devices:
            item.add_marker(pytest.mark.skip(reason=f"test requires one of {devices} (current: {CURRENT_DEVICE})"))


def _handle_slow_tests(items: list[Item]):
    """Skip slow tests unless RUN_SLOW is enabled."""
    if not is_env_enabled("RUN_SLOW", "0"):
        skip_slow = pytest.mark.skip(reason="slow test (set RUN_SLOW=1 to run)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def _get_visible_devices_env() -> str | None:
    """Return device visibility env var name."""
    if CURRENT_DEVICE == "cuda":
        return "CUDA_VISIBLE_DEVICES"
    if CURRENT_DEVICE == "npu":
        return "ASCEND_RT_VISIBLE_DEVICES"
    return None


def _handle_device_visibility(items: list[Item]):
    """Handle device visibility based on test markers.

    - If NO test has @require_distributed:
        -> force single-device visibility
    - If ANY test has @require_distributed:
        -> allow multi-device visibility
        -> but skip tests whose required device count is not met.
    """
    env_key = _get_visible_devices_env()
    if env_key is None or CURRENT_DEVICE == "cpu":
        return

    # Parse visible devices
    visible_devices_env = os.environ.get(env_key)
    if visible_devices_env:
        visible_devices = [v for v in visible_devices_env.split(",") if v != ""]
    else:
        visible_devices = []

    has_distributed_test = any(item.get_closest_marker("require_distributed") is not None for item in items)

    # -------------------------------
    # Case 1: no distributed tests
    # -------------------------------
    if not has_distributed_test:
        # hard lock to single device
        if visible_devices:
            os.environ[env_key] = visible_devices[0]
        else:
            os.environ[env_key] = "0"
        return

    # -------------------------------
    # Case 2: distributed tests exist
    # -------------------------------
    available = len(visible_devices) if visible_devices else 1

    for item in items:
        marker = item.get_closest_marker("require_distributed")
        if not marker:
            continue

        required = marker.args[0] if marker.args else 2
        if available < required:
            item.add_marker(pytest.mark.skip(reason=f"test requires {required} devices, but only {available} visible"))
        else:
            os.environ[env_key] = ",".join(str(i) for i in range(required))


def pytest_collection_modifyitems(config: Config, items: list[Item]):
    """Modify test collection based on markers and environment."""
    _handle_slow_tests(items)
    _handle_runs_on(items)
    _handle_device_visibility(items)


@pytest.fixture
def fix_valuehead_cpu_loading():
    """Fix valuehead model loading."""
    patch_valuehead_model()
