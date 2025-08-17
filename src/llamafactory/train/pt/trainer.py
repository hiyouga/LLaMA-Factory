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

import importlib.util
from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import (
    check_deepspeed_fp8_compatibility,
    create_deepspeed_fp8_kwargs,
    validate_fp8_requirements,
)
from ...data.processor.alst_data_adapter import create_alst_data_adapter
from ...model.model_utils.alst_config import create_alst_config
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_sequence_parallel_group, update_alst_adapter_with_model


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomTrainer(Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 for both DeepSpeed and native Accelerate
        if model_args is not None and model_args.fp8:
            if not validate_fp8_requirements():
                logger.warning("FP8 requirements not met, falling back to default precision")
            elif not check_deepspeed_fp8_compatibility():
                logger.warning("FP8 disabled due to DeepSpeed incompatibility - please upgrade to DeepSpeed 0.17.3+")
            else:
                import os
                
                # Always set mixed precision to fp8 first
                os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
                logger.info_rank0("Set ACCELERATE_MIXED_PRECISION=fp8")
                
                # Configure FP8 backend and options
                backend = getattr(model_args, 'fp8_backend', 'auto')
                if backend != 'auto':
                    os.environ["FP8_BACKEND"] = backend
                    logger.info_rank0(f"Set FP8_BACKEND={backend}")
                
                # Create and validate recipe kwargs (for logging/debugging)
                if importlib.util.find_spec("deepspeed") is not None:
                    deepspeed_fp8_kwargs = create_deepspeed_fp8_kwargs(model_args)
                    logger.info_rank0(f"DeepSpeed FP8 kwargs created: {deepspeed_fp8_kwargs}")
                else:
                    fp8_kwargs = create_fp8_kwargs(model_args)
                    logger.info_rank0(f"Native FP8 kwargs created: {fp8_kwargs}")
                    
                    if hasattr(model_args, 'fp8_enable_fsdp_float8_all_gather') and model_args.fp8_enable_fsdp_float8_all_gather:
                        os.environ["FP8_ENABLE_FSDP_FLOAT8_ALL_GATHER"] = "true"
                        logger.info_rank0("Set FP8_ENABLE_FSDP_FLOAT8_ALL_GATHER=true")
                
                logger.info_rank0("FP8 environment variables configured for Accelerate")

        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
            
        # Initialize ALST data adapter if needed
        if model_args is not None and model_args.sequence_parallel_size > 1:
            self.alst_config = create_alst_config(model_args)
            if self.alst_config.enabled:
                # Get sequence parallel group from model
                sp_group = getattr(model, 'sequence_parallel_group', None)
                self.alst_data_adapter = create_alst_data_adapter(model_args, self.alst_config, sp_group)
                logger.info_rank0("ALST data adapter initialized for trainer")
            else:
                self.alst_data_adapter = None
        else:
            self.alst_data_adapter = None

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
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)
    
    @override
    def get_train_dataloader(self) -> "torch.utils.data.DataLoader":
        """Override to add ALST DataLoader wrapping if enabled."""
        dataloader = super().get_train_dataloader()
        
        if self.alst_data_adapter is not None:
            # Update ALST adapter with sequence parallel group from model
            update_alst_adapter_with_model(self.alst_data_adapter, self.model, self.accelerator)
            
            # Estimate sequence length from dataset if possible
            sequence_length = getattr(self.args, 'max_seq_length', None) or getattr(self.train_dataset, 'max_length', None)
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
            sequence_length = getattr(self.args, 'max_seq_length', None) or getattr(eval_dataset or self.eval_dataset, 'max_length', None)
            dataloader = self.alst_data_adapter.wrap_dataloader(dataloader, sequence_length)
            logger.info_rank0("Applied ALST wrapping to evaluation DataLoader")
            
        return dataloader
