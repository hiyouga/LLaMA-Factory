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
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Optional, Union

from omegaconf import OmegaConf
from transformers import HfArgumentParser

from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.config.sample_args import SampleArguments
from llamafactory.v1.config.training_args import TrainingArguments
from llamafactory.v1.config.validators import run_cross_validators


@dataclass(frozen=True)
class RuntimeArgs:
    """Aggregate container for V1 arguments (minimal API)."""

    data: Optional[DataArguments] = None
    model: Optional[ModelArguments] = None
    training: Optional[TrainingArguments] = None
    sample: Optional[SampleArguments] = None

    def __post_init__(self) -> None:
        """Auto-validate on construction."""
        # 1) Call per-dataclass validate() if available
        for sec in (self.data, self.model, self.training, self.sample):
            if sec is None:
                continue
            validator = getattr(sec, "validate", None)
            if callable(validator):
                validator()
        
        # 2) Run registered cross-section validators
        run_cross_validators(self)

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
        """Explicit validation (already done in __post_init__, but kept for compatibility)."""
        # Already validated in __post_init__, this is a no-op but kept for explicit calls
        pass


def _prepare_args(
    args: Optional[Union[dict[str, Any], list[str]]] = None,
) -> Union[dict[str, Any], list[str]]:
    """Prepare raw args from CLI/config file."""
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
    return args


def _parse_selected_dataclasses(
    dataclass_types: list[type],
    args: Optional[Union[dict[str, Any], list[str]]] = None,
) -> tuple:
    """Parse only selected dataclass types."""
    parser = HfArgumentParser(dataclass_types)
    # Allow extra keys since we're only parsing a subset
    allow_extra_keys = True

    args = _prepare_args(args)

    if isinstance(args, dict):
        (*parsed_args,) = parser.parse_dict(args, allow_extra_keys=allow_extra_keys)
    else:
        (*parsed_args, _unknown_args) = parser.parse_args_into_dataclasses(args, return_remaining_strings=True)
        # Ignore unknown args when parsing subset

    return tuple(parsed_args)


_CACHED_TRAINING_RUNTIME_ARGS: RuntimeArgs | None = None
_CACHED_EVAL_RUNTIME_ARGS: RuntimeArgs | None = None


def get_training_runtime_args(
    args: Optional[Union[dict[str, Any], list[str]]] = None,
    *,
    refresh: bool = False,
) -> RuntimeArgs:
    """Get RuntimeArgs containing only model+training sections."""
    global _CACHED_TRAINING_RUNTIME_ARGS
    if not refresh and args is None and _CACHED_TRAINING_RUNTIME_ARGS is not None:
        return _CACHED_TRAINING_RUNTIME_ARGS
    
    # Parse only needed dataclasses
    model_args, training_args = _parse_selected_dataclasses(
        [ModelArguments, TrainingArguments],
        args
    )
    
    # Construct RuntimeArgs (will auto-validate in __post_init__)
    runtime = RuntimeArgs(
        model=model_args,
        training=training_args,
    )
    _CACHED_TRAINING_RUNTIME_ARGS = runtime
    return runtime


def get_eval_runtime_args(
    args: Optional[Union[dict[str, Any], list[str]]] = None,
    *,
    refresh: bool = False,
) -> RuntimeArgs:
    """Get RuntimeArgs containing only model+data sections."""
    global _CACHED_EVAL_RUNTIME_ARGS
    if not refresh and args is None and _CACHED_EVAL_RUNTIME_ARGS is not None:
        return _CACHED_EVAL_RUNTIME_ARGS
    
    # Parse only needed dataclasses
    data_args, model_args = _parse_selected_dataclasses(
        [DataArguments, ModelArguments],
        args
    )
    
    # Construct RuntimeArgs (will auto-validate in __post_init__)
    runtime = RuntimeArgs(
        data=data_args,
        model=model_args,
    )
    _CACHED_EVAL_RUNTIME_ARGS = runtime
    return runtime


if __name__ == "__main__":
    print(get_training_runtime_args())
