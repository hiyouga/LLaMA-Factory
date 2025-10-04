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
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import OmegaConf
from transformers import HfArgumentParser

from .data_args import DataArguments
from .model_args import ModelArguments
from .sample_args import SampleArguments
from .training_args import TrainingArguments


def get_args(
    args: Optional[Union[dict[str, Any], list[str]]] = None,
) -> tuple[DataArguments, ModelArguments, TrainingArguments, SampleArguments]:
    parser = HfArgumentParser([DataArguments, ModelArguments, TrainingArguments, SampleArguments])

    if args is None:
        if sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml") or sys.argv[1].endswith(".json"):
            override_config = OmegaConf.from_cli(sys.argv[2:])
            dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
            args = OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
        else:  # list of strings
            args = sys.argv[1:]

    if isinstance(args, dict):
        (*parsed_args, unknown_args) = parser.parse_dict(args, allow_extra_keys=True)
    else:
        (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(args, return_remaining_strings=True)
