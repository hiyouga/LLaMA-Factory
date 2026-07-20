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

import pytest
import torch.multiprocessing as mp
from torch import nn
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


def test_flash_linear_attention_kernels_compose_with_auto(monkeypatch):
    import fsdp_turbo.ops.fla as fla_ops

    from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_default_kernels, get_default_kernels
    from llamafactory.v1.plugins.model_plugins.kernels.ops.linear_attention.fla import _FlashLinearAttentionOpKernel

    model = nn.Sequential(nn.Linear(2, 2))
    selected = "fused_recurrent_gated_delta_rule, chunk_gated_delta_rule"
    calls = []

    assert set(selected.replace(" ", "").split(",")).issubset(get_default_kernels())
    monkeypatch.setattr(_FlashLinearAttentionOpKernel, "check_deps", classmethod(lambda cls: True))
    monkeypatch.setattr(
        fla_ops,
        "apply_fla_ops",
        lambda received_model, received_names, strict: calls.append((received_model, received_names, strict)) or 1,
    )

    assert apply_default_kernels(model, include_kernels=selected) is model
    assert calls == [
        (model, ["fused_recurrent_gated_delta_rule"], False),
        (model, ["chunk_gated_delta_rule"], False),
    ]


def test_flash_linear_attention_plugin_uses_strict_standard_kernels(monkeypatch):
    from llamafactory.v1.plugins.model_plugins.kernels import interface

    model = nn.Sequential(nn.Linear(2, 2))
    calls = []
    monkeypatch.setattr(
        interface,
        "apply_kernel",
        lambda kernel, **kwargs: calls.append((kernel, kwargs)),
    )

    assert interface.apply_flash_linear_attention_kernels(model, include_kernels="auto") is model
    assert calls == [
        ("chunk_gated_delta_rule", {"model": model, "strict": True}),
        ("fused_recurrent_gated_delta_rule", {"model": model, "strict": True}),
    ]

    with pytest.raises(ValueError, match="Unsupported Flash Linear Attention kernels"):
        interface.apply_flash_linear_attention_kernels(model, include_kernels="not_a_kernel")
