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

from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoModelForCausalLM

from llamafactory.v1.accelerator.helper import get_current_accelerator
from llamafactory.v1.plugins.model_plugins.kernels.mlp import npu_swiglu
from llamafactory.v1.plugins.model_plugins.kernels.registry import apply_available_kernels, apply_kernel
from llamafactory.v1.plugins.model_plugins.kernels.rms_norm import npu_rms_norm
from llamafactory.v1.plugins.model_plugins.kernels.rope import npu_rope


@pytest.fixture(autouse=True)
def clear_accelerator_cache():
    get_current_accelerator.cache_clear()


@patch("torch.accelerator.current_accelerator")
def test_apply_kernel(mock_get_accelerator: MagicMock):
    mock_device = MagicMock()
    setattr(mock_device, "type", "npu")
    mock_get_accelerator.return_value = mock_device

    model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen2.5")

    original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
    original_swiglu_forward = model.model.layers[0].mlp.forward

    apply_kernel(model, npu_rope.NpuRoPEKernel)

    model = apply_kernel(model, npu_rms_norm.NpuRMSNormKernel)
    assert model.model.layers[0].input_layernorm is not original_rmsnorm_forward

    model = apply_kernel(model, npu_swiglu.NpuSwiGluKernel)
    assert model.model.layers[0].mlp.forward is not original_swiglu_forward


@patch("torch.accelerator.current_accelerator")
def test_apply_all_kernels(mock_get_accelerator: MagicMock):
    get_current_accelerator.cache_clear()
    mock_device = MagicMock()
    setattr(mock_device, "type", "npu")
    mock_get_accelerator.return_value = mock_device

    model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen2.5")

    original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
    original_swiglu_forward = model.model.layers[0].mlp.forward

    model = apply_available_kernels(model)

    assert model.model.layers[0].input_layernorm is not original_rmsnorm_forward
    assert model.model.layers[0].mlp.forward is not original_swiglu_forward
