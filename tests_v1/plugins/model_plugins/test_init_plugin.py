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


from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.config.arg_parser import get_args
from llamafactory.v1.core.model_engine import ModelEngine


def test_init_on_meta():
    model_args, *_ = get_args(
        dict(
            model="llamafactory/tiny-random-qwen3",
            init_config={"name": "init_on_meta"},
        )
    )
    model_engine = ModelEngine(model_args=model_args)
    assert model_engine.model.device.type == "meta"


def test_init_on_rank0():
    model_args, *_ = get_args(
        dict(
            model="llamafactory/tiny-random-qwen3",
            init_config={"name": "init_on_rank0"},
        )
    )
    model_engine = ModelEngine(model_args=model_args)
    if DistributedInterface().get_rank() == 0:
        assert model_engine.model.device.type == "cpu"
    else:
        assert model_engine.model.device.type == "meta"


def test_init_on_default():
    model_args, *_ = get_args(
        dict(
            model="llamafactory/tiny-random-qwen3",
            init_config={"name": "init_on_default"},
        )
    )
    model_engine = ModelEngine(model_args=model_args)
    assert model_engine.model.device == DistributedInterface().current_device


if __name__ == "__main__":
    """
    python tests_v1/plugins/model_plugins/test_init_plugin.py
    """
    test_init_on_meta()
    test_init_on_rank0()
    test_init_on_default()
