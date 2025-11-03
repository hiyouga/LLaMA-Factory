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

"""Cross-section validators for V1 arguments.

Register validators here to check conflicts between different argument sections.
"""

from dataclasses import fields as dataclass_fields
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from llamafactory.v1.config.parser import RuntimeArgs


# Registry for cross-section validators: {frozenset of section names: validator_fn}
_CROSS_VALIDATORS: dict[frozenset[str], list[Callable[["RuntimeArgs"], None]]] = {}


def register_cross_validator(sections: tuple[str, ...], validator: Callable[["RuntimeArgs"], None]) -> None:
    """Register a cross-section validator.
    
    Args:
        sections: Tuple of section names, e.g., ("model", "training") or ("data", "model", "training")
        validator: Callable that takes RuntimeArgs and raises on validation failure
        
    Example:
        def validate_model_training(rt: RuntimeArgs) -> None:
            if rt.model.some_field conflicts with rt.training.some_field:
                raise ValueError("...")
        
        register_cross_validator(("model", "training"), validate_model_training)
    """
    key = frozenset(sections)
    if key not in _CROSS_VALIDATORS:
        _CROSS_VALIDATORS[key] = []
    _CROSS_VALIDATORS[key].append(validator)


def run_cross_validators(runtime: "RuntimeArgs") -> None:
    """Run registered validators only when all required sections are present."""
    # Dynamically check which sections are present (not None)
    present = {
        field.name 
        for field in dataclass_fields(runtime) 
        if getattr(runtime, field.name) is not None
    }
    
    # Run validators only if all their required sections exist
    for required_sections, validators in _CROSS_VALIDATORS.items():
        if required_sections.issubset(present):
            for validator in validators:
                validator(runtime)


def validate_model_training_conflict(runtime: "RuntimeArgs") -> None:
    """Example: validate model and training arguments don't conflict."""
    model = runtime.model
    training = runtime.training
    print(model)
    assert model.model is not None
    pass


def validate_data_model_training_conflict(runtime: "RuntimeArgs") -> None:
    """Example: validate across data, model, and training."""
    data = runtime.data
    model = runtime.model
    training = runtime.training

    pass


# 注册所有跨段校验器
# 注意：这些注册会在模块导入时执行
register_cross_validator(("model", "training"), validate_model_training_conflict)
register_cross_validator(("data", "model", "training"), validate_data_model_training_conflict)

