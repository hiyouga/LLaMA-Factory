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

from ..config.model_args import ModelArguments
from ..extras.types import Model, Processor


class ModelEngine:
    def __init__(self, model_args: ModelArguments) -> None:
        self.args = model_args

    def get_model(self) -> Model:
        pass

    def get_processor(self) -> Processor:
        pass
