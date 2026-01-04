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

from ..config import ModelArguments, SampleArguments, SampleBackend
from ..utils.types import HFModel, Processor, TorchDataset


class BaseEngine(ABC):
    @abstractmethod
    def __init__(
        self,
        args: SampleArguments,
        model_args: ModelArguments,
        model: HFModel = None,
        processor: Processor = None,
    ) -> None:
        """Initialize the engine.

        Args:
            args: Sample arguments.
            model_args: Model arguments.
            model: Model.
            processor: Processor.
        """
        ...

    @abstractmethod
    async def generate(self, messages):
        pass

    @abstractmethod
    async def batch_infer(self, data: TorchDataset) -> None:
        pass


class HuggingFaceEngine(BaseEngine):
    def __init__(
        self,
        args: SampleArguments,
        model_args: ModelArguments,
        model: HFModel,
        processor: Processor,
    ) -> None:
        self.args = args


class BaseSampler:
    def __init__(
        self,
        args: SampleArguments,
        model_args: ModelArguments,
        model: HFModel,
        processor: Processor,
    ) -> None:
        if args.sample_backend == SampleBackend.HF:
            self.engine = HuggingFaceEngine(args, model_args, model, processor)
        else:
            raise ValueError(f"Unknown sample backend: {args.sample_backend}")

    async def generate(self, messages):
        return await self.engine.generate(messages)

    async def batch_infer(self, data: TorchDataset) -> None:
        return await self.engine.batch_infer(data)
