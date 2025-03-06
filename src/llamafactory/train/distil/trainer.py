# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

from copy import deepcopy
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomDistillationTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self,
        teacher_model: Union["PreTrainedModel", torch.nn.Module],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        **kwargs,
    ):
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)

        if teacher_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(teacher_model, "is_loaded_in_8bit", False)
                    or getattr(teacher_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.teacher_model = self._prepare_deepspeed(teacher_model)
            else:
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        labels = inputs.get("labels")
        padding_mask = labels.eq(-100)
        label_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        # Shape: (batch_size, seq_len, vocab_size)
        teacher_prob = torch.nn.functional.softmax(
            teacher_outputs.logits / self.finetuning_args.distilling_temperature, dim=-1
        )
        student_logprob = torch.nn.functional.log_softmax(
            outputs.logits / self.finetuning_args.distilling_temperature, dim=-1
        )
        kl_losses = (teacher_prob * (teacher_prob.log() - student_logprob)).sum(dim=-1)
        kl_losses.masked_fill_(padding_mask, 0)
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        loss = (
            self.finetuning_args.distilling_lambda
            * kl_losses.mean()
            / (num_active_elements * student_logprob.shape[-1])
            + label_loss
        )

        if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
            loss = loss / self.args.gradient_accumulation_steps

        return (loss, outputs) if return_outputs else loss

    def _prepare_deepspeed(self, model: "PreTrainedModel"):
        import deepspeed  # type: ignore

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
