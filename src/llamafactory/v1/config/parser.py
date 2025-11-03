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

from transformers.hf_argparser import DataClass


import json
import sys
from pathlib import Path
from dataclasses import dataclass, replace, fields as dataclass_fields
from typing import Any, Optional, Union

from omegaconf import OmegaConf
from transformers import HfArgumentParser

from llamafactory.extras.misc import is_env_enabled
from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.config.sample_args import SampleArguments
from llamafactory.v1.config.training_args import TrainingArguments


T = None  # placeholder to avoid unused imports after simplification


@dataclass(frozen=True)
class RuntimeArgs:
    """Aggregate container for V1 arguments (minimal API)."""

    data: DataArguments
    model: ModelArguments
    training: TrainingArguments
    sample: SampleArguments

    def __repr__(self) -> str:  # pragma: no cover - concise & future-proof
        parts = []
        for f in dataclass_fields(self):
            value = getattr(self, f.name, None)
            # Try to summarize as ClassName(fields=N) when value is a dataclass
            try:
                n = len(dataclass_fields(value))  # type: ignore[arg-type]
                parts.append(f"{f.name}={value.__class__.__name__}(fields={n})")
            except TypeError:
                parts.append(f"{f.name}={type(value).__name__}")
        return "RuntimeArgs(" + ", ".join(parts) + ")"

    def validate(self) -> None:
        """System-level coupled validation across sections.

        Notes:
            Keep this method focused on cross-section constraints and core sanity
            checks. Per-section deep validation (if any) can be implemented on
            each dataclass and optionally invoked here when present.
        """

        # 1) Per-section basic sanity checks
        data = self.data
        model = self.model
        train = self.training
        sample = self.sample

        #TODO
        #Do validation at runtime parse called



def _parse_dataclasses(
    args: Optional[Union[dict[str, Any], list[str]]] = None,
) -> tuple[DataArguments, ModelArguments, TrainingArguments, SampleArguments]:
    """Internal: parse arguments from command line or config file into dataclasses."""
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

    return tuple[DataClass, ...](parsed_args)


def parse_args(
    args: Optional[Union[dict[str, Any], list[str]]] = None,
) -> RuntimeArgs:
    """Public entrypoint: parse and aggregate into a single RuntimeArgs container."""

    data_args, model_args, training_args, sample_args = _parse_dataclasses(args)
    runtime = RuntimeArgs(
        data=data_args,
        model=model_args,
        training=training_args,
        sample=sample_args,
    )
    # Auto-validate once when args become effective
    runtime.validate()
    return runtime


_CACHED_RUNTIME_ARGS: RuntimeArgs | None = None


def get_runtime_args(
    args: Optional[Union[dict[str, Any], list[str]]] = None,
    *,
    refresh: bool = False,
) -> RuntimeArgs:
    """Get parsed RuntimeArgs with optional caching.

    - If refresh is False and cached value exists (and args is None), returns cached.
    - Otherwise parses (optionally with provided args) and updates cache.
    """
    global _CACHED_RUNTIME_ARGS
    if not refresh and args is None and _CACHED_RUNTIME_ARGS is not None:
        return _CACHED_RUNTIME_ARGS

    runtime = parse_args(args)
    _CACHED_RUNTIME_ARGS = runtime
    return runtime


if __name__ == "__main__":
    breakpoint()
    print(get_runtime_args().training)
