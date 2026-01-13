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

import os
import sys
import traceback

import pytest
import torch.multiprocessing as mp


# Ensure repo root is on sys.path so torch mp spawn can import `tests_v1.*`
# during pickling/unpickling (important for pytest --import-mode=importlib).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture(scope="module", autouse=True)
def suppress_tokenizers_parallelism_warning():
    """Suppress tokenizers parallelism warning when spawning."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _subprocess_wrapper(target_func, queue):
    """Wrapper function for subprocess execution (must be at module level for pickling)."""
    try:
        target_func()
        queue.put((True, None))
    except Exception as e:
        queue.put((False, f"{e}\n{traceback.format_exc()}"))


def _run_in_subprocess(target_func):
    """Run a function in a subprocess to isolate module state (torch mp spawn)."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    proc = ctx.Process(target=_subprocess_wrapper, args=(target_func, queue))
    proc.start()
    proc.join(timeout=1200)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise RuntimeError("Subprocess timed out and was terminated.")
    elif proc.exitcode != 0:
        raise RuntimeError(f"Subprocess exited with code {proc.exitcode}")

    success, error = queue.get()
    if not success:
        raise AssertionError(error)


def _reload_kernels():
    """Helper to reload kernel modules to respect mocked accelerator."""
    keys_to_remove = [k for k in sys.modules if k.startswith("llamafactory.v1.plugins.model_plugins.kernels")]
    for k in keys_to_remove:
        del sys.modules[k]


def _test_apply_kernel_impl():
    """Actual test logic for apply_kernel, runs in subprocess."""
    from unittest.mock import MagicMock, patch

    from transformers import AutoModelForCausalLM

    from llamafactory.v1.accelerator.helper import get_current_accelerator

    get_current_accelerator.cache_clear()

    with patch("torch.accelerator.current_accelerator") as mock_get_accelerator:
        mock_device = MagicMock()
        setattr(mock_device, "type", "npu")
        mock_get_accelerator.return_value = mock_device

        _reload_kernels()
        from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_default_kernels

        model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen3")
        original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
        original_swiglu_forward = model.model.layers[0].mlp.forward

        model = apply_default_kernels(model=model, include_kernels="npu_fused_rmsnorm")

        assert model.model.layers[0].input_layernorm.forward.__func__ is not original_rmsnorm_forward.__func__
        assert model.model.layers[0].mlp.forward.__func__ is original_swiglu_forward.__func__


def _test_apply_all_kernels_impl():
    """Actual test logic for apply_all_kernels, runs in subprocess."""
    from unittest.mock import MagicMock, patch

    from transformers import AutoModelForCausalLM

    from llamafactory.v1.accelerator.helper import get_current_accelerator

    get_current_accelerator.cache_clear()

    with patch("torch.accelerator.current_accelerator") as mock_get_accelerator:
        mock_device = MagicMock()
        setattr(mock_device, "type", "npu")
        mock_get_accelerator.return_value = mock_device

        _reload_kernels()
        from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_default_kernels

        model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen3")
        original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
        original_swiglu_forward = model.model.layers[0].mlp.forward

        model = apply_default_kernels(model=model, include_kernels=True)

        assert model.model.layers[0].input_layernorm.forward.__func__ is not original_rmsnorm_forward.__func__
        assert model.model.layers[0].mlp.forward.__func__ is not original_swiglu_forward.__func__


def test_apply_kernel():
    """Test applying a specific kernel (runs in isolated subprocess)."""
    _run_in_subprocess(_test_apply_kernel_impl)


def test_apply_all_kernels():
    """Test applying all kernels (runs in isolated subprocess)."""
    _run_in_subprocess(_test_apply_all_kernels_impl)