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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

import llamafactory.train.ppo.trainer as ppo_trainer


class FailingModel:
    def generate(self, **kwargs):
        raise RuntimeError("generation failed")

    def __call__(self, **kwargs):
        raise RuntimeError("reward failed")


def test_get_inputs_restores_layernorm_after_generation_error(monkeypatch):
    model = FailingModel()
    restore_layernorm = Mock()
    trainer = SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(unwrap_model=lambda _: model),
        model_args=SimpleNamespace(upcast_layernorm=True),
        generation_config=object(),
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=1),
    )
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
    }
    layernorm_params = {"layernorm.weight": torch.ones(1)}
    monkeypatch.setattr(ppo_trainer, "unwrap_model_for_generation", lambda *_: nullcontext(model))
    monkeypatch.setattr(ppo_trainer, "dump_layernorm", Mock(return_value=layernorm_params))
    monkeypatch.setattr(ppo_trainer, "restore_layernorm", restore_layernorm)

    with pytest.raises(RuntimeError, match="generation failed"):
        ppo_trainer.CustomPPOTrainer.get_inputs(trainer, batch)

    restore_layernorm.assert_called_once_with(model, layernorm_params)


def test_get_rewards_restores_default_adapter_after_forward_error(monkeypatch):
    model = FailingModel()
    replace_model = Mock()
    trainer = SimpleNamespace(
        model=model,
        reward_model=None,
        accelerator=SimpleNamespace(unwrap_model=lambda _: model),
        finetuning_args=SimpleNamespace(reward_model_type="lora"),
        amp_context=nullcontext(),
        prepare_model_inputs=lambda *_: {"attention_mask": torch.ones((1, 2), dtype=torch.long)},
    )
    monkeypatch.setattr(ppo_trainer, "unwrap_model_for_generation", lambda *_: nullcontext())
    monkeypatch.setattr(ppo_trainer, "replace_model", replace_model)

    with pytest.raises(RuntimeError, match="reward failed"):
        ppo_trainer.CustomPPOTrainer.get_rewards(trainer, [torch.tensor([1])], [torch.tensor([2])])

    assert replace_model.call_args_list == [call(model, target="reward"), call(model, target="default")]
