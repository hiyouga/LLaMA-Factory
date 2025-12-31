# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/training_args.py
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
from enum import Enum, unique


class PluginConfig(dict):
    """Dictionary that allows attribute access."""

    @property
    def name(self) -> str:
        """Plugin name."""
        if "name" not in self:
            raise ValueError("Plugin configuration must have a 'name' field.")

        return self["name"]


PluginArgument = PluginConfig | dict | str | None


@unique
class ModelClass(str, Enum):
    """Auto class for model config."""

    LLM = "llm"
    CLS = "cls"
    OTHER = "other"


@unique
class SampleBackend(str, Enum):
    HF = "hf"
    VLLM = "vllm"


def _convert_str_dict(data: dict) -> dict:
    """Parse string representation inside the dictionary.

    Args:
        data: The string or dictionary to convert.

    Returns:
        The converted dictionary.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            if value.lower() in ("true", "false"):
                data[key] = value.lower() == "true"
            elif value.isdigit():
                data[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                data[key] = float(value)

    return data


def get_plugin_config(config: PluginArgument) -> PluginConfig | None:
    """Get the plugin configuration from the argument value.

    Args:
        config: The argument value to get the plugin configuration from.

    Returns:
        The plugin configuration.
    """
    if config is None:
        return None

    if isinstance(config, str) and config.startswith("{"):
        config = json.loads(config)

    config = _convert_str_dict(config)
    if "name" not in config:
        raise ValueError("Plugin configuration must have a 'name' field.")

    return PluginConfig(config)
