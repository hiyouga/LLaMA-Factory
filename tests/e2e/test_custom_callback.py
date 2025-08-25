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
from transformers import TrainerCallback

from llamafactory.train.tuner import run_exp


class TestFlagCallback(TrainerCallback):
    __test__ = False

    def __init__(self, flag_file=None):
        super().__init__()
        self.flag_file = flag_file or "/tmp/callback_test_flag.txt"

    def on_train_begin(self, args, state, control, **kwargs):
        with open(self.flag_file, "w") as f:
            f.write("callback_triggered")


def test_custom_callback_triggers(tmp_path):
    flag_file = tmp_path / "callback_test_flag.txt"
    if flag_file.exists():
        flag_file.unlink()

    args = {
        "model_name_or_path": "llamafactory/tiny-random-Llama-3",
        "do_train": True,
        "finetuning_type": "lora",
        "dataset": "llamafactory/tiny-supervised-dataset",
        "dataset_dir": "ONLINE",
        "template": "llama3",
        "cutoff_len": 1,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "report_to": "none",
        "output_dir": str(tmp_path / "output"),
        "callbacks": [
            {"name": "tests.e2e.test_custom_callback.TestFlagCallback", "args": {"flag_file": str(flag_file)}}
        ],
    }
    run_exp(args)
    assert flag_file.exists()
    assert flag_file.read_text() == "callback_triggered"
