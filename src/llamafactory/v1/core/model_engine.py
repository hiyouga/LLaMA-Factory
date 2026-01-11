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

"""The definition of model engine.

How to use:
model_engine = ModelEngine(model_args, is_train=True)
model_engine.processor: Get the tokenizer or multi-modal processor.
model_engine.renderer: Get the renderer.
model_engine.model_config: Get the model configuration.
model_engine.model: Get the HF model.

Init workflow:
1. Init processor.
2. Init render.
2. Init model config.
3. Init model.
4. Init adapter.
"""

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoProcessor

from ..accelerator.helper import DeviceType
from ..accelerator.interface import DistributedInterface
from ..config.model_args import ModelArguments, ModelClass
from ..utils import logging
from ..utils.types import HFConfig, HFModel, Processor
from .utils.rendering import Renderer


logger = logging.get_logger(__name__)


class ModelEngine:
    """Model engine.

    Args:
        model_args: Model arguments.
        is_train: Whether to train the model.
    """

    def __init__(self, model_args: ModelArguments, is_train: bool = False) -> None:
        self.args = model_args
        """Model arguments."""
        self.is_train = is_train
        """Whether to train the model."""
        self.processor = self._init_processor()
        """Tokenizer or multi-modal processor."""
        self.renderer = Renderer(self.args.template, self.processor)
        """Renderer."""
        self.model_config = self._init_model_config()
        """Model configuration."""
        self.model = self._init_model()
        """HF model."""

    def _init_processor(self) -> Processor:
        """Init processor.

        NOTE: Transformers v5 always use fast tokenizer.
        https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/auto/tokenization_auto.py#L642
        """
        return AutoProcessor.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
        )

    def _init_model_config(self) -> HFConfig:
        """Init model config."""
        return AutoConfig.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
        )

    def _init_model(self) -> HFModel:
        """Init model.

        Transformers can choose the proper model init context.
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

        if self.args.init_config is not None:
            from ..plugins.model_plugins.initialization import InitPlugin

            init_device = InitPlugin(self.args.init_config.name)()
        else:
            init_device = DistributedInterface().current_device

        if init_device.type == DeviceType.META:
            with init_empty_weights():
                model = AutoClass.from_config(self.model_config)
        else:
            model = AutoClass.from_pretrained(
                self.args.model,
                config=self.model_config,
                dtype="auto",
                device_map=init_device,
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

        if self.args.kernel_config is not None:
            from ..plugins.model_plugins.kernels.interface import KernelPlugin

            model = KernelPlugin(self.args.kernel_config.name)(
                model, include_kernels=self.args.kernel_config.get("include_kernels")
            )

        return model


if __name__ == "__main__":
    """
    python -m llamafactory.v1.core.model_engine --model llamafactory/tiny-random-qwen2.5
    """
    from ..config.arg_parser import get_args

    model_args, *_ = get_args()
    model_engine = ModelEngine(model_args=model_args)
    print(model_engine.processor)
    print(model_engine.model_config)
    print(model_engine.model)
