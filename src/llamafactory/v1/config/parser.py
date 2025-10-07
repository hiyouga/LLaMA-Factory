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


import json
import sys
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import OmegaConf
from transformers import HfArgumentParser

from ...extras.misc import is_env_enabled
from .data_args import DataArguments
from .model_args import ModelArguments
from .sample_args import SampleArguments
from .training_args import TrainingArguments


def get_args(
    args: Optional[Union[dict[str, Any], list[str]]] = None,
) -> tuple[DataArguments, ModelArguments, TrainingArguments, SampleArguments]:
    """Parse arguments from command line or config file."""
    parser = HfArgumentParser([DataArguments, ModelArguments, TrainingArguments, SampleArguments])
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_KEYS")

    if args is None:
        if len(sys.argv) > 1 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
            override_config = OmegaConf.from_cli(sys.argv[2:])
            dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
            args = OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
        elif len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
            override_config = OmegaConf.from_cli(sys.argv[2:])
            dict_config = OmegaConf.create(json.load(Path(sys.argv[1]).absolute()))
            args = OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
        else:  # list of strings
            args = sys.argv[1:]

    if isinstance(args, dict):
        (*parsed_args,) = parser.parse_dict(args, allow_extra_keys=allow_extra_keys)
    else:
        (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(args, return_remaining_strings=True)
        if unknown_args and not allow_extra_keys:
            print(parser.format_help())
            print(f"Got unknown args, potentially deprecated arguments: {unknown_args}")
            raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {unknown_args}")

    return tuple(parsed_args)


if __name__ == "__main__":
    print(get_args())
