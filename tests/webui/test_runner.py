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

from llamafactory.webui import runner as runner_module
from llamafactory.webui.locales import ALERTS
from llamafactory.webui.runner import Runner


class _Manager:
    def get_elem_by_id(self, elem_id):
        return elem_id


def test_check_output_dir_without_saved_config(monkeypatch, tmp_path):
    runner = Runner(_Manager())
    monkeypatch.setattr(runner_module, "get_save_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(runner_module.gr, "Warning", lambda *args, **kwargs: None)

    output = runner.check_output_dir("en", "model", "lora", "output")

    assert output == {"train.output_box": ALERTS["warn_output_dir_exists"]["en"]}


def test_check_output_dir_restores_saved_config(monkeypatch, tmp_path):
    runner = Runner(_Manager())
    monkeypatch.setattr(runner_module, "get_save_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(runner_module, "load_args", lambda path: {"train.learning_rate": 1e-5})
    monkeypatch.setattr(runner_module.gr, "Warning", lambda *args, **kwargs: None)

    output = runner.check_output_dir("en", "model", "lora", "output")

    assert output["train.learning_rate"] == 1e-5
