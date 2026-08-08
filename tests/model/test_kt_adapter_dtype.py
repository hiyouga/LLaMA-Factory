# Copyright 2026 the LlamaFactory team.
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

from types import SimpleNamespace

import pytest
import torch

from llamafactory.hparams import ModelArguments
from llamafactory.model import adapter as adapter_module


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="dummy")


def _finetuning_args(**overrides):
    values = {
        "finetuning_type": "lora",
        "lora_target": ["proj"],
        "use_llama_pro": False,
        "use_dora": False,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "use_rslora": False,
        "additional_target": None,
        "pissa_init": False,
        "create_new_adapter": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_new_kt_adapter_preserves_bf16(monkeypatch):
    captured_kwargs = {}
    model = _DummyModel()
    monkeypatch.setattr(adapter_module, "patch_target_modules", lambda model, args, modules: modules)
    monkeypatch.setattr(adapter_module, "LoraConfig", lambda **kwargs: kwargs)

    def fake_get_peft_model(base_model, peft_config, **kwargs):
        captured_kwargs.update(kwargs)
        return base_model

    monkeypatch.setattr(adapter_module, "get_peft_model", fake_get_peft_model)
    adapter_module._setup_lora_tuning(
        config=model.config,
        model=model,
        model_args=ModelArguments(model_name_or_path="dummy", use_kt=True),
        finetuning_args=_finetuning_args(),
        is_trainable=True,
        cast_trainable_params_to_fp32=False,
    )
    assert captured_kwargs == {"autocast_adapter_dtype": False}


def test_non_kt_adapter_keeps_peft_default(monkeypatch):
    captured_kwargs = {}
    model = _DummyModel()
    monkeypatch.setattr(adapter_module, "patch_target_modules", lambda model, args, modules: modules)
    monkeypatch.setattr(adapter_module, "LoraConfig", lambda **kwargs: kwargs)

    def fake_get_peft_model(base_model, peft_config, **kwargs):
        captured_kwargs.update(kwargs)
        return base_model

    monkeypatch.setattr(adapter_module, "get_peft_model", fake_get_peft_model)
    adapter_module._setup_lora_tuning(
        config=model.config,
        model=model,
        model_args=ModelArguments(model_name_or_path="dummy"),
        finetuning_args=_finetuning_args(),
        is_trainable=True,
        cast_trainable_params_to_fp32=False,
    )
    assert captured_kwargs == {}


@pytest.mark.parametrize("is_trainable", [True, False])
def test_loaded_kt_adapter_preserves_saved_dtype(monkeypatch, is_trainable):
    captured_kwargs = {}
    model = _DummyModel()
    model_args = ModelArguments(
        model_name_or_path="dummy",
        adapter_name_or_path="adapter",
        use_kt=True,
    )
    if not is_trainable:
        model_args._kt_adapter_artifact_path = "adapter"
        monkeypatch.setattr(adapter_module, "_load_kt_inference_adapter_artifacts", lambda model, path: None)

    def fake_from_pretrained(base_model, adapter_path, **kwargs):
        captured_kwargs.update(kwargs)
        return base_model

    monkeypatch.setattr(adapter_module.PeftModel, "from_pretrained", fake_from_pretrained)
    adapter_module._setup_lora_tuning(
        config=model.config,
        model=model,
        model_args=model_args,
        finetuning_args=_finetuning_args(),
        is_trainable=is_trainable,
        cast_trainable_params_to_fp32=False,
    )
    assert captured_kwargs["autocast_adapter_dtype"] is False
    assert captured_kwargs["is_trainable"] is is_trainable
