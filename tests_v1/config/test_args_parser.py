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

import pathlib
import sys
from unittest.mock import patch

from llamafactory.v1.config.arg_parser import get_args


def test_get_args_from_yaml(tmp_path: pathlib.Path):
    config_yaml = """
        ### model
        model: "llamafactory/tiny-random-qwen2.5"
        trust_remote_code: true
        use_fast_processor: true
        model_class: "llm"
        kernel_config:
          name: "auto"
          include_kernels: "auto" # choice: null/true/false/auto/kernel_id1,kernel_id2,kernel_id3, default is null
        peft_config:
          name: "lora"
          lora_rank: 0.8
        quant_config: null

        ### data
        dataset: "llamafactory/tiny-supervised-dataset"
        cutoff_len: 2048

        ### training
        output_dir: "outputs/test_run"
        micro_batch_size: 1
        global_batch_size: 1
        learning_rate: 1.0e-4
        bf16: false
        dist_config: null

        ### sample
        sample_backend: "hf"
        max_new_tokens: 128
    """

    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml, encoding="utf-8")

    test_argv = ["test_args_parser.py", str(config_file)]

    with patch.object(sys, "argv", test_argv):
        data_args, model_args, training_args, sample_args = get_args()
        assert training_args.output_dir == "outputs/test_run"
        assert training_args.micro_batch_size == 1
        assert training_args.global_batch_size == 1
        assert training_args.learning_rate == 1.0e-4
        assert training_args.bf16 is False
        assert training_args.dist_config is None
        assert model_args.model == "llamafactory/tiny-random-qwen2.5"
        assert model_args.kernel_config.name == "auto"
        assert model_args.kernel_config.get("include_kernels") == "auto"
        assert model_args.peft_config.name == "lora"
        assert model_args.peft_config.get("lora_rank") == 0.8
