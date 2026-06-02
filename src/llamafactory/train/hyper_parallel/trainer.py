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

"""HyperParallel distributed trainer for LlamaFactory."""

import logging
import os
import types
from contextlib import nullcontext
from typing import Any, Optional

import torch
from hyper_parallel.integration.llamafactory import (
    HSDPModule,
    HyperParallelArguments,
    export_to_hf_format,
    fsdp2_prepare_model,
    hsdp_sync_stream,
    load_hsdp_model,
    load_hsdp_optimizer_and_scheduler,
    save_hsdp_checkpoint,
    wrap_optimizer_with_skip_dtensor_dispatch,
)
from hyper_parallel.integration.llamafactory import (
    clip_grad_norm_ as hp_clip_grad_norm_,
)
from torch import nn

from ..sft.trainer import CustomSeq2SeqTrainer


logger = logging.getLogger(__name__)


class HyperParallelTrainer(CustomSeq2SeqTrainer):
    """Trainer that replaces Accelerate FSDP2 with HyperParallel fully_shard.

    Inherits CustomSeq2SeqTrainer for training algorithm logic (loss, metrics,
    prediction, sampler, etc.) and only overrides HSDP-specific behavior.
    """

    def __init__(
        self,
        hp_args: HyperParallelArguments,
        finetuning_args=None,
        processor=None,
        ref_model: Optional[nn.Module] = None,
        **kwargs,
    ):
        self._hp_args = hp_args

        # Let CustomSeq2SeqTrainer handle everything except ref_model —
        # Custom would prepare it with accelerate's fsdp2_prepare_model,
        # but we need HP's version instead.
        super().__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            ref_model=None,
            **kwargs,
        )

        if not getattr(self.accelerator, "is_fsdp2", False):
            raise ValueError("HyperParallel trainer requires Accelerate FSDP2 mode to be enabled.")

        # Prepare ref_model with HP's fsdp2_prepare_model
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model, self._hp_args)

        self._orig_accelerator_clip_grad_norm = self.accelerator.clip_grad_norm_
        self._orig_fsdp2_prepare_model = None
        self._accelerator_patches_active = False

    def _activate_accelerator_patches(self) -> None:
        """Patch Accelerate to use HyperParallel fsdp2_prepare_model and clip_grad_norm_."""
        if self._accelerator_patches_active:
            return

        import accelerate.accelerator as acc_module  # pylint: disable=C0415

        hp_args = self._hp_args

        self._orig_fsdp2_prepare_model = acc_module.fsdp2_prepare_model

        def _hp_fsdp2_prepare_model(accelerator, model):
            return fsdp2_prepare_model(accelerator, model, hp_args)

        acc_module.fsdp2_prepare_model = _hp_fsdp2_prepare_model

        def _hp_clip_grad_norm(accelerator, parameters, max_norm, norm_type=2):
            if getattr(accelerator, "is_fsdp2", False):
                accelerator.unscale_gradients()
                parameter_list = list(parameters)
                parameter_ids = {id(param) for param in parameter_list}
                for model in accelerator._models:  # pylint: disable=protected-access
                    if not isinstance(model, HSDPModule):
                        continue
                    model_param_ids = {id(param) for param in model.parameters()}
                    if parameter_ids and parameter_ids.issubset(model_param_ids):
                        return hp_clip_grad_norm_(parameter_list, max_norm, norm_type=norm_type)
            return self._orig_accelerator_clip_grad_norm(parameters, max_norm, norm_type=norm_type)

        self.accelerator.clip_grad_norm_ = types.MethodType(_hp_clip_grad_norm, self.accelerator)
        self._accelerator_patches_active = True

    def _restore_accelerator_patches(self) -> None:
        """Restore original Accelerate methods."""
        if not self._accelerator_patches_active:
            return

        import accelerate.accelerator as acc_module  # pylint: disable=C0415

        if self._orig_fsdp2_prepare_model is not None:
            acc_module.fsdp2_prepare_model = self._orig_fsdp2_prepare_model
        self.accelerator.clip_grad_norm_ = self._orig_accelerator_clip_grad_norm
        self._accelerator_patches_active = False

    def _wrap_model(self, model: nn.Module, training: bool = True, dataloader=None) -> nn.Module:
        """Let Accelerate own FSDP2/HSDP wrapping so optimizer remapping stays correct."""
        del dataloader
        if isinstance(model, HSDPModule):
            return model
        if training and getattr(self.accelerator, "is_fsdp2", False):
            return model
        return super()._wrap_model(model, training=training)

    def _move_model_to_device(self, model: nn.Module, device: Optional[torch.device] = None):
        """Skip redundant device moves for HSDP-wrapped models."""
        if isinstance(model, HSDPModule):
            return model
        if device is None:
            return model
        return model.to(device)

    def train(self, *args, **kwargs):
        """Activate HP patches during training and restore afterwards."""
        self._activate_accelerator_patches()
        try:
            return super().train(*args, **kwargs)
        finally:
            self._restore_accelerator_patches()

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Standard training step with HSDP gradient synchronization."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        sync_gradients = getattr(self.accelerator, "sync_gradients", True)
        if isinstance(model, HSDPModule):
            model.set_is_last_backward(sync_gradients)
            model.set_requires_gradient_sync(sync_gradients)

        compute_loss_context_manager = getattr(self, "compute_loss_context_manager", nullcontext)
        with compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if not getattr(self, "model_accepts_loss_kwargs", False) and getattr(self, "compute_loss_func", None) is None:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        if isinstance(model, HSDPModule) and sync_gradients:
            hsdp_sync_stream()

        return loss.detach()

    def create_optimizer(self):
        """Create optimizer and wrap step with SkipDTensorDispatch."""
        optimizer = super().create_optimizer()
        wrap_optimizer_with_skip_dtensor_dispatch(optimizer)
        return optimizer

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        """Save model/optimizer shards per-rank and scheduler."""
        save_hsdp_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            output_dir=output_dir,
            should_save_scheduler=self.args.should_save and self.lr_scheduler is not None,
        )

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Optional[nn.Module] = None) -> None:
        """Load model from HSDP sharded checkpoint."""
        target = model if model is not None else self.model
        loaded = load_hsdp_model(target, resume_from_checkpoint)
        if not loaded:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        self._pending_hsdp_checkpoint = resume_from_checkpoint
        return None

    def _load_optimizer_and_scheduler(self, checkpoint: Optional[str] = None) -> None:
        """Load optimizer/scheduler from per-rank checkpoint files."""
        ckpt_dir = getattr(self, "_pending_hsdp_checkpoint", None) or checkpoint
        if ckpt_dir is None:
            return
        load_hsdp_optimizer_and_scheduler(self.optimizer, self.lr_scheduler, ckpt_dir)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model weights in HuggingFace-compatible format."""
        save_dir = output_dir or self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        export_to_hf_format(self.model, getattr(self, "processing_class", None), save_dir)
