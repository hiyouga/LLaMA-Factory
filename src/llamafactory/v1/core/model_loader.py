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

"""The definition of model loader.

Init Phase:
1. Init processor.
2. Init model config.
3. Init model.
4. Init adapter.

"""

import torch
from transformers import AutoConfig, AutoProcessor

from ..accelerator.interface import DistributedInterface
from ..config.model_args import ModelArguments, ModelClass
from ..utils import logging
from ..utils.types import HFConfig, HFModel, Processor


logger = logging.get_logger(__name__)


class ModelLoader:
    """Model loader.

    Args:
        model_args: Model arguments.
        is_trainable: Whether to train the model.
    """

    def __init__(self, model_args: ModelArguments, is_train: bool = False) -> None:
        self.args = model_args
        """Model arguments."""
        self.is_train = is_train
        """Whether to train the model."""
        self.processor = self._init_processor()
        """Tokenizer or multi-modal processor."""
        self.model_config = self._init_model_config()
        """Model configuration."""
        self.model = self._init_model()
        """HF model."""

    def _init_processor(self) -> Processor:
        """Init processor."""
        return AutoProcessor.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
            use_fast=self.args.use_fast_processor,
        )

    def _init_model_config(self) -> HFConfig:
        """Init model config."""
        return AutoConfig.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
        )

    def _init_model(self) -> HFModel:
        """Init model.

        Let transformers handle the model init context.
        https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/modeling_utils.py#L3538
        """
        if self.args.model_class == ModelClass.LLM:
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

            if type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
                AutoClass = AutoModelForImageTextToText
            else:
                AutoClass = AutoModelForCausalLM

        elif self.args.model_class == ModelClass.CLS:
            from transformers import AutoModelForTokenClassification

            AutoClass = AutoModelForTokenClassification
        else:
            from transformers import AutoModel

            AutoClass = AutoModel

        # map the entire model to the current accelerator
        model = AutoClass.from_pretrained(
            self.args.model,
            config=self.model_config,
            dtype="auto",
            device_map=DistributedInterface.current_accelerator,
            trust_remote_code=self.args.trust_remote_code,
        )

        if self.args.peft_config is None:
            if self.is_train:
                logger.info_rank0("Fine-tuning mode: full tuning")
                model = model.to(torch.float32)
            else:
                logger.info_rank0("Inference the original model")
        else:
            from ..plugins.model_plugins.peft import PeftPlugin

            model = PeftPlugin(self.args.peft_config.name)(model, self.args.peft_config, self.is_train)

        return model


if __name__ == "__main__":
    """
    python -m llamafactory.v1.core.model_loader --model llamafactory/tiny-random-qwen2.5
    """
    from ..config.arg_parser import get_args

    _, model_args, *_ = get_args()
    model_loader = ModelLoader(model_args=model_args)
    print(model_loader.processor)
    print(model_loader.model_config)
    print(model_loader.model)
