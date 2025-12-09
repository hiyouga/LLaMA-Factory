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

import torch
from transformers import AutoConfig, AutoProcessor

from ..accelerator.helper import DeviceType
from ..config.model_args import AutoClass, ModelArguments
from ..utils.types import HFConfig, HFModel, Processor


class ModelWorker:
    def __init__(self, model_args: ModelArguments) -> None:
        self.args = model_args
        """Model arguments."""
        self.processor: Optional[Processor] = None
        """Tokenizer or multi-modal processor."""
        self.model_config: Optional[HFConfig] = None
        """Model configuration."""
        self.model: Optional[HFModel] = None
        """HF model."""
        self.is_adapter = False
        """Whether the model has adapter."""

    def init_processor(self) -> None:
        if self.processor is not None:
            return

        self.processor = AutoProcessor.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
            use_fast=self.args.use_fast_processor,
        )

    def init_model_config(self) -> None:
        if self.model_config is not None:
            return

        self.model_config = AutoConfig.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
        )

    def init_model(self) -> None:
        if self.model is not None:
            return

        self.init_model_config()

        if self.args.auto_class == AutoClass.CAUSALLM:
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

            if type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
                ModelClass = AutoModelForImageTextToText
            else:
                ModelClass = AutoModelForCausalLM
        elif self.args.auto_class == AutoClass.CLASSIFICATION:
            from transformers import AutoModelForTokenClassification

            ModelClass = AutoModelForTokenClassification
        else:
            from transformers import AutoModel

            ModelClass = AutoModel

        default_device_type = torch.get_default_device().type
        if default_device_type == DeviceType.META:
            self.model = ModelClass.from_config(self.model_config)
        else:
            self.model = ModelClass.from_pretrained(
                self.args.model,
                config=self.model_config,
                dtype="auto",
                device_map=default_device_type,
                trust_remote_code=self.args.trust_remote_code,
            )

    def init_adapter(self) -> None:
        if self.is_adapter:
            return

        if self.args.peft_config is not None:
            from ..plugins.model_plugins.peft import PeftPlugin

            self.model = PeftPlugin(self.args.peft_config.name)(self.model, self.args.peft_config)

        self.is_adapter = True

    def get_processor(self) -> Processor:
        return self.processor

    def get_model_config(self) -> HFConfig:
        return self.model_config

    def get_model(self) -> HFModel:
        return self.model
