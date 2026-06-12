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
from functools import partial
from typing import Any, Optional

import torch
import torch.distributed as dist
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
from hyper_parallel.integration.llamafactory.context_parallel import (
    cp_prepare_model,
    get_cp_group,
    get_cp_group_ranks,
    get_cp_rank,
    get_dp_rank,
    shard_inputs_for_cp,
)
from hyper_parallel.platform import get_platform
from torch import nn

from ..sft.trainer import CustomSeq2SeqTrainer


logger = logging.getLogger(__name__)


class _CPBatchRepeatedBatchSampler(torch.utils.data.BatchSampler):
    """Repeat logical batches so Accelerate shards CP peers onto the same samples."""

    def __init__(self, sampler, batch_size: int, drop_last: bool, repeat_factor: int):
        super().__init__(sampler, batch_size, drop_last)
        self.repeat_factor = repeat_factor

    def __len__(self):
        return super().__len__() * self.repeat_factor

    def __iter__(self):
        for batch in super().__iter__():
            for _ in range(self.repeat_factor):
                yield list(batch)


class _CPDataLoaderLengthProxy:
    """Keep baseline logical dataloader length while yielding CP-repeated batches."""

    def __init__(self, dataloader, logical_length: int):
        self._dataloader = dataloader
        self._logical_length = logical_length

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return self._logical_length

    def __getattr__(self, name):
        return getattr(self._dataloader, name)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _broadcast_input_value(value, src_rank: int, group, default_device: torch.device):
    """Broadcast one nested input value from the CP source rank to all peers."""
    current_rank = get_platform().get_rank()
    if isinstance(value, torch.Tensor):
        metadata = [(tuple(value.shape), value.dtype, value.device.type)] if current_rank == src_rank else [None]
        dist.broadcast_object_list(metadata, src=src_rank, group=group)
        shape, dtype, device_type = metadata[0]
        if current_rank != src_rank:
            device = torch.device("cpu") if device_type == "cpu" else default_device
            value = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(value, src=src_rank, group=group)
        return value

    if isinstance(value, dict):
        keys = [list(value.keys())] if current_rank == src_rank else [None]
        dist.broadcast_object_list(keys, src=src_rank, group=group)
        source_dict = value if current_rank == src_rank else {}
        return {key: _broadcast_input_value(source_dict.get(key), src_rank, group, default_device) for key in keys[0]}

    if isinstance(value, list):
        length = [len(value)] if current_rank == src_rank else [None]
        dist.broadcast_object_list(length, src=src_rank, group=group)
        source_list = value if current_rank == src_rank else [None] * length[0]
        return [_broadcast_input_value(source_list[idx], src_rank, group, default_device) for idx in range(length[0])]

    if isinstance(value, tuple):
        length = [len(value)] if current_rank == src_rank else [None]
        dist.broadcast_object_list(length, src=src_rank, group=group)
        source_tuple = value if current_rank == src_rank else (None,) * length[0]
        return tuple(_broadcast_input_value(source_tuple[idx], src_rank, group, default_device) for idx in range(length[0]))

    payload = [value] if current_rank == src_rank else [None]
    dist.broadcast_object_list(payload, src=src_rank, group=group)
    return payload[0]


def _synchronize_inputs_across_cp_group(
    inputs: dict[str, Any],
    cp_group,
    cp_group_ranks: Optional[tuple[int, ...]],
    default_device: torch.device,
) -> dict[str, Any]:
    """Force all ranks in one CP group to consume the same prepared inputs."""
    if cp_group is None or not cp_group_ranks or not dist.is_available() or not dist.is_initialized():
        return inputs
    return _broadcast_input_value(inputs, cp_group_ranks[0], cp_group, default_device)


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

        self._cp_size = hp_args.cp_size
        self._cp_rank = get_cp_rank(hp_args) if self._cp_size > 1 else 0
        self._dp_rank = get_dp_rank(hp_args) if self._cp_size > 1 else get_platform().get_rank()
        self._cp_group = get_cp_group(hp_args) if self._cp_size > 1 else None
        self._cp_group_ranks = get_cp_group_ranks(hp_args) if self._cp_size > 1 else None

        # Prepare ref_model with the same CP + HSDP path as the train model.
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model = self._prepare_model_for_hyper_parallel(self.ref_model)

        self._orig_accelerator_clip_grad_norm = self.accelerator.clip_grad_norm_
        self._orig_fsdp2_prepare_model = None
        self._accelerator_patches_active = False

    def _prepare_model_for_hyper_parallel(self, model: nn.Module) -> nn.Module:
        """Apply CP runtime hooks before delegating to HyperParallel FSDP2 preparation."""
        if self._cp_size > 1:
            model = cp_prepare_model(model, self.accelerator, self._hp_args)
        return fsdp2_prepare_model(self.accelerator, model, self._hp_args)

    def _activate_accelerator_patches(self) -> None:
        """Patch Accelerate to use HyperParallel fsdp2_prepare_model and clip_grad_norm_."""
        if self._accelerator_patches_active:
            return

        import accelerate.accelerator as acc_module  # pylint: disable=C0415

        self._orig_fsdp2_prepare_model = acc_module.fsdp2_prepare_model

        def _hp_fsdp2_prepare_model(accelerator, model):
            return self._prepare_model_for_hyper_parallel(model)

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

    def _get_train_sampler(self, train_dataset=None):
        """Match the no-CP baseline sampler semantics before CP repeats whole logical batches."""
        if train_dataset is None:
            train_dataset = self.train_dataset
        if getattr(self.finetuning_args, "disable_shuffling", False):
            return torch.utils.data.SequentialSampler(train_dataset)
        return super()._get_train_sampler(train_dataset)

    def _build_cp_batch_sampler(self, dataset, shuffle: bool, batch_size: int, drop_last: bool):
        """Repeat complete logical batches so CP groups consume the same baseline batch."""
        sampler = self._get_train_sampler(dataset) if shuffle else torch.utils.data.SequentialSampler(dataset)
        return _CPBatchRepeatedBatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            repeat_factor=self._cp_size,
        )

    def _get_cp_dataloader(self, dataset, batch_size: int, shuffle: bool):
        """Create a train dataloader whose logical batches are shared within each CP group."""
        if isinstance(dataset, torch.utils.data.IterableDataset):
            return super()._get_dataloader(
                dataset=dataset,
                description="Training",
                batch_size=batch_size,
                sampler_fn=None,
                is_training=True,
            )

        try:
            import datasets  # pylint: disable=C0415
        except ImportError:  # pragma: no cover
            datasets = None

        if datasets is not None and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description="Training")
            data_collator = self.data_collator
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description="Training")

        batch_sampler = self._build_cp_batch_sampler(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=self.args.dataloader_drop_last,
        )
        logical_batches = len(batch_sampler) // self._cp_size
        dp_size = max(1, get_platform().get_world_size() // self._cp_size)
        logical_length = logical_batches // dp_size if self.args.dataloader_drop_last else _ceil_div(logical_batches, dp_size)

        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        from transformers.trainer import seed_worker  # pylint: disable=C0415

        dataloader_params["worker_init_fn"] = partial(
            seed_worker,
            num_workers=self.args.dataloader_num_workers,
            rank=self.args.process_index,
        )

        dataloader = self.accelerator.prepare(torch.utils.data.DataLoader(dataset, **dataloader_params))
        return _CPDataLoaderLengthProxy(dataloader, logical_length)

    def get_train_dataloader(self):
        """Keep the no-CP logical batch stream, then repeat each whole batch across CP peers."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self._cp_size <= 1:
            return super().get_train_dataloader()

        shuffle = not getattr(self.finetuning_args, "disable_shuffling", False)
        return self._get_cp_dataloader(
            dataset=self.train_dataset,
            batch_size=self._train_batch_size,
            shuffle=shuffle,
        )

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
        """Standard training step with HSDP sync plus optional CP input sharding."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self._cp_size > 1:
            inputs = _synchronize_inputs_across_cp_group(
                inputs,
                self._cp_group,
                self._cp_group_ranks,
                self.accelerator.device,
            )
            inputs = shard_inputs_for_cp(inputs, self._cp_rank, self._cp_size)

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
