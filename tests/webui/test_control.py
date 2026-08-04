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

import json

import pytest

from llamafactory.extras.constants import TRAINER_LOG
from llamafactory.webui import control


COMPLETE_LOG = {
    "current_steps": 1,
    "total_steps": 2,
    "percentage": 50,
    "elapsed_time": "1s",
    "remaining_time": "1s",
}


def test_get_trainer_info_ignores_incomplete_trailing_log(tmp_path, monkeypatch):
    trainer_log = tmp_path / TRAINER_LOG
    trainer_log.write_text(json.dumps(COMPLETE_LOG) + '\n{"current_steps":', encoding="utf-8")
    monkeypatch.setattr(control.gr, "Slider", lambda **kwargs: kwargs)

    _, progress, _ = control.get_trainer_info("en", tmp_path, do_train=False)

    assert progress["value"] == 50
    assert progress["visible"] is True


def test_get_trainer_info_rejects_complete_invalid_log(tmp_path, monkeypatch):
    trainer_log = tmp_path / TRAINER_LOG
    trainer_log.write_text(json.dumps(COMPLETE_LOG) + "\nnot-json\n", encoding="utf-8")
    monkeypatch.setattr(control.gr, "Slider", lambda **kwargs: kwargs)

    with pytest.raises(json.JSONDecodeError):
        control.get_trainer_info("en", tmp_path, do_train=False)
