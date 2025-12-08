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

"""The definition of model worker.

Init Phase:
1. Init processor.
2. Init model config.
3. Init model.
4. Init adapter.

"""

from typing import Optional

from transformers import AutoConfig, AutoProcessor

from ..config.model_args import ModelArguments
from ..extras.types import DistModel, HFConfig, HFModel, Processor


class ModelWorker:
    def __init__(self, model_args: ModelArguments) -> None:
        self.args = model_args
        """Model arguments."""
        self.processor: Optional[Processor] = None
        """Tokenizer or multi-modal processor."""
        self.model_config: Optional[HFConfig] = None
        """Model configuration."""
        self.unwrapped_model: Optional[HFModel] = None
        """Unwrapped model."""
        self.model: Optional[DistModel] = None
        """Distributed model."""
        self.init_processor()
        self.init_model_config()
        self.init_model()
        self.init_adapter()

    def init_processor(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
            use_fast=self.args.use_fast_processor,
        )

    def init_model_config(self) -> None:
        self.model_config = AutoConfig.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
        )

    def init_model(self) -> None:
        if self.args.auto_model_class == "causallm":
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

            if type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
                AutoClass = AutoModelForImageTextToText
            else:
                AutoClass = AutoModelForCausalLM
        elif self.args.auto_model_class == "classification":
            from transformers import AutoModelForTokenClassification

            AutoClass = AutoModelForTokenClassification
        else:
            from transformers import AutoModel

            AutoClass = AutoModel

        self.unwrapped_model = AutoClass.from_pretrained(
            self.args.model,
            config=self.model_config,
            dtype="auto",
            device_map="cpu",
            trust_remote_code=self.args.trust_remote_code,
        )

    def init_adapter(self) -> None:
        pass

    def get_processor(self) -> Processor:
        return self.processor

    def get_model_config(self) -> HFConfig:
        return self.model_config

    def get_model(self) -> HFModel:
        return self.unwrapped_model
