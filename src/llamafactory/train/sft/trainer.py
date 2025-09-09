# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...data.processor.alst_data_adapter import create_alst_data_adapter
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ...model.model_utils.alst_config import create_alst_config
from ..alst_loss import create_alst_loss_handler, should_use_alst_loss
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import (
    create_custom_optimizer,
    create_custom_scheduler,
    get_sequence_parallel_group,
    update_alst_adapter_with_model,
)


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)

        # Synchronize gradient accumulation steps between Accelerate and DeepSpeed/Training args
        training_args = kwargs.get("args")
        if training_args is not None and hasattr(training_args, "gradient_accumulation_steps"):
            import os

            gradient_accumulation_steps = training_args.gradient_accumulation_steps
            os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(gradient_accumulation_steps)
            logger.info_rank0(f"Set ACCELERATE_GRADIENT_ACCUMULATION_STEPS={gradient_accumulation_steps}")

        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self.model_args = model_args  # Store for FP8 logging
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # Initialize ALST data adapter and loss handler if needed
        if model_args is not None and model_args.sequence_parallel_size > 1:
            self.alst_config = create_alst_config(model_args)
            if self.alst_config.enabled:
                # Get sequence parallel group from model (will be None initially, set later)
                self.alst_data_adapter = create_alst_data_adapter(model_args, self.alst_config, None)
                self.alst_loss_handler = None  # Will be created when sequence parallel group is available
                logger.info_rank0("ALST data adapter initialized for trainer")
            else:
                self.alst_data_adapter = None
                self.alst_loss_handler = None
        else:
            self.alst_data_adapter = None
            self.alst_loss_handler = None

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Optional Cut Cross-Entropy support for memory-efficient loss
        if finetuning_args.use_cce:
            from ...extras.packages import is_cce_available

            if not is_cce_available():
                raise ImportError(
                    "Cut Cross-Entropy is not available. Install with: "
                    "pip install 'cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git'"
                )
            from ..trainer_utils import cce_loss_func

            self.compute_loss_func = cce_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

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
    def training_step(self, model, inputs, *args, **kwargs):
        # ALST doesn't require dummy forward pass - UlyssesSPDataLoaderAdapter handles sequence parallel setup
        return super().training_step(model, inputs, *args, **kwargs)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        # ALST/UlyssesSPDataLoaderAdapter handles data distribution - use standard sampler logic
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""Compute loss with ALST support for sequence parallel training."""
        sequence_parallel_group = get_sequence_parallel_group(model, self.accelerator)

        # Check if we should use ALST loss computation
        if should_use_alst_loss(inputs, sequence_parallel_group):
            # Initialize ALST loss handler if not already done
            if self.alst_loss_handler is None:
                self.alst_loss_handler = create_alst_loss_handler(sequence_parallel_group)
            # Use ALST loss computation
            return self.alst_loss_handler.compute_alst_loss(model, inputs, return_outputs)

        else:
            # Standard training (no sequence parallelism) or ALST not properly configured
            if sequence_parallel_group is not None:
                logger.warning(
                    f"Sequence parallel group exists but ALST loss not triggered. Input keys: {list(inputs.keys())}"
                )

            loss = super().compute_loss(model, inputs, return_outputs, **kwargs)

        if is_transformers_version_greater_than("4.46") and not getattr(self, "model_accepts_loss_kwargs", False):
            # other model should not scale the loss
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    @override
    def get_train_dataloader(self) -> "torch.utils.data.DataLoader":
        """Override to add ALST DataLoader wrapping if enabled."""
        dataloader = super().get_train_dataloader()

        if self.alst_data_adapter is not None:
            # Update ALST adapter with sequence parallel group from model
            update_alst_adapter_with_model(self.alst_data_adapter, self.model, self.accelerator)

            # Estimate sequence length from dataset if possible
            sequence_length = getattr(self.args, "max_seq_length", None) or getattr(
                self.train_dataset, "max_length", None
            )
            dataloader = self.alst_data_adapter.wrap_dataloader(dataloader, sequence_length)
            logger.info_rank0("Applied ALST wrapping to training DataLoader")

        return dataloader

    @override
    def get_eval_dataloader(self, eval_dataset=None) -> "torch.utils.data.DataLoader":
        """Override to add ALST DataLoader wrapping if enabled."""
        dataloader = super().get_eval_dataloader(eval_dataset)

        if self.alst_data_adapter is not None:
            # Update ALST adapter with sequence parallel group from model
            update_alst_adapter_with_model(self.alst_data_adapter, self.model, self.accelerator)

            # Estimate sequence length from dataset if possible
            sequence_length = getattr(self.args, "max_seq_length", None) or getattr(
                eval_dataset or self.eval_dataset, "max_length", None
            )
            dataloader = self.alst_data_adapter.wrap_dataloader(dataloader, sequence_length)
            logger.info_rank0("Applied ALST wrapping to evaluation DataLoader")

        return dataloader
