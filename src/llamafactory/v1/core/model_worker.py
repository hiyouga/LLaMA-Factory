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

from transformers import AutoConfig, AutoProcessor

from ..config.model_args import ModelArguments
from ..extras.types import HFConfig, HFModel, Processor


class ModelWorker:
    def __init__(self, model_args: ModelArguments) -> None:
        self.args = model_args
        """Model arguments."""

    def get_processor(self) -> Processor:
        return AutoProcessor.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
            use_fast=self.args.use_fast_processor,
        )

    def get_model_config(self) -> HFConfig:
        return AutoConfig.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
        )

    def get_model(self, model_config: HFConfig) -> HFModel:
        if self.args.auto_model_class == "causallm":
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

            if type(model_config) in AutoModelForImageTextToText._model_mapping.keys():
                AutoClass = AutoModelForImageTextToText
            else:
                AutoClass = AutoModelForCausalLM
        elif self.args.auto_model_class == "classification":
            from transformers import AutoModelForTokenClassification

            AutoClass = AutoModelForTokenClassification
        else:
            from transformers import AutoModel

            AutoClass = AutoModel

        return AutoClass.from_pretrained(
            self.args.model,
            config=model_config,
            dtype="auto",
            device_map="cpu",
            trust_remote_code=self.args.trust_remote_code,
        )
