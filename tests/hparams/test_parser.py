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
from transformers.training_args import ParallelMode

from llamafactory.hparams import parser
from llamafactory.hparams.parser import _normalize_swanlab_args


@pytest.mark.parametrize(
    ("report_to", "use_swanlab", "expected_report_to", "expected_use_swanlab"),
    [
        (["swanlab"], False, [], True),
        (["tensorboard", "swanlab", "wandb"], False, ["tensorboard", "wandb"], True),
        (["swanlab"], True, [], True),
        (["tensorboard"], False, ["tensorboard"], False),
    ],
)
def test_normalize_swanlab_args(report_to, use_swanlab, expected_report_to, expected_use_swanlab):
    training_args = SimpleNamespace(report_to=report_to)
    finetuning_args = SimpleNamespace(use_swanlab=use_swanlab)

    _normalize_swanlab_args(training_args, finetuning_args)

    assert training_args.report_to == expected_report_to
    assert finetuning_args.use_swanlab is expected_use_swanlab


def test_report_to_swanlab_uses_native_callback(monkeypatch, tmp_path):
    monkeypatch.setattr(parser.TrainingArguments, "parallel_mode", property(lambda self: ParallelMode.DISTRIBUTED))
    monkeypatch.setattr(parser, "check_version", lambda *args, **kwargs: None)

    _, _, training_args, finetuning_args, _ = parser.get_train_args(
        {
            "model_name_or_path": "dummy",
            "output_dir": str(tmp_path),
            "report_to": "swanlab",
        }
    )

    assert training_args.report_to == []
    assert finetuning_args.use_swanlab is True
