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


@pytest.fixture(autouse=True)
def clear_accelerator_cache():
    get_current_accelerator.cache_clear()


@patch("torch.accelerator.current_accelerator")
def test_apply_kernel(mock_get_accelerator: MagicMock):
    mock_device = MagicMock()
    setattr(mock_device, "type", "npu")
    mock_get_accelerator.return_value = mock_device

    from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_default_kernels

    model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen2.5")
    original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
    original_swiglu_forward = model.model.layers[0].mlp.forward
    model = apply_default_kernels(model=model, use_v1_kernels="npu_fused_rmsnorm")
    assert model.model.layers[0].input_layernorm.forward.__func__ is not original_rmsnorm_forward.__func__
    assert model.model.layers[0].mlp.forward.__func__ is original_swiglu_forward.__func__


@patch("torch.accelerator.current_accelerator")
def test_apply_all_kernels(mock_get_accelerator: MagicMock):
    get_current_accelerator.cache_clear()
    mock_device = MagicMock()
    setattr(mock_device, "type", "npu")
    mock_get_accelerator.return_value = mock_device

    from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_default_kernels

    model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen2.5")

    original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
    original_swiglu_forward = model.model.layers[0].mlp.forward

    model = apply_default_kernels(model=model, use_v1_kernels=True)
    assert model.model.layers[0].input_layernorm.forward.__func__ is not original_rmsnorm_forward.__func__
    assert model.model.layers[0].mlp.forward.__func__ is not original_swiglu_forward.__func__
