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

from llamafactory.v1.config.parser import RuntimeArgs, register_cross_validator


def validate_model_training_conflict(runtime: RuntimeArgs) -> None:
    """Example: validate model and training arguments don't conflict."""
    model = runtime.model
    training = runtime.training
    
    pass


def validate_data_model_training_conflict(runtime: RuntimeArgs) -> None:
    """Example: validate across data, model, and training."""
    data = runtime.data
    model = runtime.model
    training = runtime.training
    
    pass


# 注册所有跨段校验器
# 注意：这些注册会在模块导入时执行
register_cross_validator(("model", "training"), validate_model_training_conflict)
register_cross_validator(("data", "model", "training"), validate_data_model_training_conflict)

