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

import sys
from unittest.mock import MagicMock, patch

import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM


def _apply_kernel(rank) -> None:
    with patch("torch.accelerator.current_accelerator") as mock_get_accelerator:
        mock_device = MagicMock()
        setattr(mock_device, "type", "npu")
        mock_get_accelerator.return_value = mock_device

        # reload kernel modules to respect mocked accelerator
        for k in list(sys.modules.keys()):
            if k.startswith("llamafactory.v1.plugins.model_plugins.kernels"):
                del sys.modules[k]

        from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_default_kernels

        model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen3")
        original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
        original_swiglu_forward = model.model.layers[0].mlp.forward

        model = apply_default_kernels(model=model, include_kernels="npu_fused_rmsnorm")

        assert model.model.layers[0].input_layernorm.forward.__func__ is not original_rmsnorm_forward.__func__
        assert model.model.layers[0].mlp.forward.__func__ is original_swiglu_forward.__func__


def _apply_all_kernels(rank) -> None:
    with patch("torch.accelerator.current_accelerator") as mock_get_accelerator:
        mock_device = MagicMock()
        setattr(mock_device, "type", "npu")
        mock_get_accelerator.return_value = mock_device

        # reload kernel modules to respect mocked accelerator
        for k in list(sys.modules.keys()):
            if k.startswith("llamafactory.v1.plugins.model_plugins.kernels"):
                del sys.modules[k]

        from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_default_kernels

        model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen3")
        original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
        original_swiglu_forward = model.model.layers[0].mlp.forward

        model = apply_default_kernels(model=model, include_kernels=True)

        assert model.model.layers[0].input_layernorm.forward.__func__ is not original_rmsnorm_forward.__func__
        assert model.model.layers[0].mlp.forward.__func__ is not original_swiglu_forward.__func__


def test_apply_kernel():
    mp.spawn(_apply_kernel)


def test_apply_all_kernels():
    mp.spawn(_apply_all_kernels)
