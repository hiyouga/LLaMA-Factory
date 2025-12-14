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

from abc import ABC, abstractmethod

from ..config.sample_args import SampleArguments, SampleBackend
from .model_loader import ModelLoader


class BaseEngine(ABC):
    @abstractmethod
    def __init__(self, sample_args: SampleArguments, model_loader: ModelLoader) -> None: ...

    @abstractmethod
    async def generate(self):
        pass

    @abstractmethod
    async def batch_infer(self):
        pass


class HuggingFaceEngine(BaseEngine):
    def __init__(self, model_loader: ModelLoader, sample_args: SampleArguments) -> None:
        self.args = sample_args


class ChatSampler:
    def __init__(self, model_loader: ModelLoader, sample_args: SampleArguments) -> None:
        if sample_args.sample_backend == SampleBackend.HF:
            self.engine = HuggingFaceEngine(model_loader, sample_args)
        else:
            raise ValueError(f"Unknown sample backend: {sample_args.sample_backend}")
