# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the original GaLore's implementation: https://github.com/jiaweizzhao/GaLore
# and the original LoRA+'s implementation: https://github.com/nikhil-ghosh-berkeley/loraplus
# and the original BAdam's implementation: https://github.com/Ledzy/BAdam
# and the HuggingFace's TRL library: https://github.com/huggingface/trl
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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from transformers import Trainer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from typing_extensions import override

from ..extras import logging
from ..extras.constants import IGNORE_INDEX, SWANLAB_CONFIG
from ..extras.packages import is_apollo_available, is_galore_available, is_ray_available
from ..hparams import FinetuningArguments, ModelArguments
from ..model import find_all_linear_modules, load_model, load_tokenizer, load_valuehead_params


if is_galore_available():
    from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit  # type: ignore


if is_apollo_available():
    from apollo_torch import APOLLOAdamW  # type: ignore


if is_ray_available():
    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TrainerCallback, TrainerState
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, RayArguments, TrainingArguments


logger = logging.get_logger(__name__)


class DummyOptimizer(torch.optim.Optimizer):
    r"""A dummy optimizer used for the GaLore or APOLLO algorithm."""

    def __init__(
        self, lr: float = 1e-3, optimizer_dict: Optional[dict["torch.nn.Parameter", "torch.optim.Optimizer"]] = None
    ) -> None:
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": lr})

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    @override
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        pass


def create_modelcard_and_push(
    trainer: "Trainer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    kwargs = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["llama-factory", finetuning_args.finetuning_type],
    }
    if data_args.dataset is not None:
        kwargs["dataset"] = data_args.dataset

    if model_args.use_unsloth:
        kwargs["tags"] = kwargs["tags"] + ["unsloth"]

    if model_args.use_kt:
        kwargs["tags"] = kwargs["tags"] + ["ktransformers"]

    if not training_args.do_train:
        pass
    elif training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(license="other", **kwargs)  # prevent from connecting to hub


def create_ref_model(
    model_args: "ModelArguments", finetuning_args: "FinetuningArguments", add_valuehead: bool = False
) -> Optional[Union["PreTrainedModel", "AutoModelForCausalLMWithValueHead"]]:
    r"""Create reference model for PPO/DPO training. Evaluation mode is not supported.

    The valuehead parameter is randomly initialized since it is useless for PPO training.
    """
    if finetuning_args.ref_model is not None:
        ref_model_args = ModelArguments.copyfrom(
            model_args,
            model_name_or_path=finetuning_args.ref_model,
            adapter_name_or_path=finetuning_args.ref_model_adapters,
            quantization_bit=finetuning_args.ref_model_quantization_bit,
        )
        ref_finetuning_args = FinetuningArguments()
        tokenizer = load_tokenizer(ref_model_args)["tokenizer"]
        ref_model = load_model(
            tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
        )
        logger.info_rank0(f"Created reference model from {finetuning_args.ref_model}")
    else:
        if finetuning_args.finetuning_type == "lora":
            ref_model = None
        else:
            ref_model_args = ModelArguments.copyfrom(model_args)
            ref_finetuning_args = FinetuningArguments()
            tokenizer = load_tokenizer(ref_model_args)["tokenizer"]
            ref_model = load_model(
                tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
            )
            logger.info_rank0("Created reference model from the model itself.")

    return ref_model


def create_reward_model(
    model: "AutoModelForCausalLMWithValueHead", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> Optional["AutoModelForCausalLMWithValueHead"]:
    r"""Create reward model for PPO training."""
    if finetuning_args.reward_model_type == "api":
        assert finetuning_args.reward_model.startswith("http"), "Please provide full url."
        logger.info_rank0(f"Use reward server {finetuning_args.reward_model}")
        return finetuning_args.reward_model
    elif finetuning_args.reward_model_type == "lora":
        model.pretrained_model.load_adapter(finetuning_args.reward_model, "reward")
        for name, param in model.named_parameters():  # https://github.com/huggingface/peft/issues/1090
            if "default" in name:
                param.data = param.data.to(torch.float32)  # trainable params should in fp32
        vhead_params = load_valuehead_params(finetuning_args.reward_model, model_args)
        assert vhead_params is not None, "Reward model is not correctly loaded."
        model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
        model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
        model.register_buffer(
            "default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False
        )
        model.register_buffer(
            "default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False
        )
        logger.info_rank0(f"Loaded adapter weights of reward model from {finetuning_args.reward_model}")
        return None
    else:
        reward_model_args = ModelArguments.copyfrom(
            model_args,
            model_name_or_path=finetuning_args.reward_model,
            adapter_name_or_path=finetuning_args.reward_model_adapters,
            quantization_bit=finetuning_args.reward_model_quantization_bit,
        )
        reward_finetuning_args = FinetuningArguments()
        tokenizer = load_tokenizer(reward_model_args)["tokenizer"]
        reward_model = load_model(
            tokenizer, reward_model_args, reward_finetuning_args, is_trainable=False, add_valuehead=True
        )
        logger.info_rank0(f"Loaded full weights of reward model from {finetuning_args.reward_model}")
        logger.warning_rank0("Please ensure the ppo model and reward model share SAME tokenizer and vocabulary.")
        return reward_model


def _get_decay_parameter_names(model: "PreTrainedModel") -> list[str]:
    r"""Return a list of names of parameters with weight decay. (weights in non-layernorm layers)."""
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def _create_galore_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    if len(finetuning_args.galore_target) == 1 and finetuning_args.galore_target[0] == "all":
        galore_targets = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
    else:
        galore_targets = finetuning_args.galore_target

    galore_params: list[torch.nn.Parameter] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(target in name for target in galore_targets):
            for param in module.parameters():
                if param.requires_grad and len(param.shape) > 1:
                    galore_params.append(param)

    galore_kwargs = {
        "rank": finetuning_args.galore_rank,
        "update_proj_gap": finetuning_args.galore_update_interval,
        "scale": finetuning_args.galore_scale,
        "proj_type": finetuning_args.galore_proj_type,
    }

    id_galore_params = {id(param) for param in galore_params}
    decay_params, nodecay_params = [], []  # they are non-galore parameters
    trainable_params: list[torch.nn.Parameter] = []  # galore_params + decay_params + nodecay_params
    decay_param_names = _get_decay_parameter_names(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            if id(param) not in id_galore_params:
                if name in decay_param_names:
                    decay_params.append(param)
                else:
                    nodecay_params.append(param)

    _, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

    if training_args.optim == "adamw_torch":
        optim_class = GaLoreAdamW
    elif training_args.optim in ["adamw_bnb_8bit", "adamw_8bit", "paged_adamw_8bit"]:
        optim_class = GaLoreAdamW8bit
    elif training_args.optim == "adafactor":
        optim_class = GaLoreAdafactor
    else:
        raise NotImplementedError(f"Unknown optim: {training_args.optim}.")

    if finetuning_args.galore_layerwise:
        logger.warning_rank0("The displayed gradient norm will be all zeros in layerwise GaLore.")
        if training_args.gradient_accumulation_steps != 1:
            raise ValueError("Per-layer GaLore does not support gradient accumulation.")

        optimizer_dict: dict[torch.Tensor, torch.optim.Optimizer] = {}
        for param in nodecay_params:
            param_groups = [dict(params=[param], weight_decay=0.0)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)
        for param in decay_params:
            param_groups = [dict(params=[param], weight_decay=training_args.weight_decay)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)
        for param in galore_params:  # galore params have weight decay
            param_groups = [dict(params=[param], weight_decay=training_args.weight_decay, **galore_kwargs)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)

        def optimizer_hook(param: "torch.nn.Parameter"):
            if param.grad is not None:
                optimizer_dict[param].step()
                optimizer_dict[param].zero_grad()

        for param in trainable_params:
            param.register_post_accumulate_grad_hook(optimizer_hook)

        optimizer = DummyOptimizer(lr=training_args.learning_rate, optimizer_dict=optimizer_dict)
    else:
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0),
            dict(params=decay_params, weight_decay=training_args.weight_decay),
            dict(params=galore_params, weight_decay=training_args.weight_decay, **galore_kwargs),
        ]
        optimizer = optim_class(param_groups, **optim_kwargs)

    logger.info_rank0(
        f"Using GaLore optimizer with args: {galore_kwargs}. "
        "It may cause hanging at the start of training, wait patiently."
    )
    return optimizer


def _create_apollo_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    if len(finetuning_args.apollo_target) == 1 and finetuning_args.apollo_target[0] == "all":
        apollo_targets = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
    else:
        apollo_targets = finetuning_args.apollo_target

    apollo_params: list[torch.nn.Parameter] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(target in name for target in apollo_targets):
            for param in module.parameters():
                if param.requires_grad and len(param.shape) > 1:
                    apollo_params.append(param)

    apollo_kwargs = {
        "rank": finetuning_args.apollo_rank,
        "proj": finetuning_args.apollo_proj,
        "proj_type": finetuning_args.apollo_proj_type,
        "update_proj_gap": finetuning_args.apollo_update_interval,
        "scale": finetuning_args.apollo_scale,
        "scale_type": finetuning_args.apollo_scale_type,
        "scale_front": finetuning_args.apollo_scale_front,
    }

    id_apollo_params = {id(param) for param in apollo_params}
    decay_params, nodecay_params = [], []  # they are non-apollo parameters
    trainable_params: list[torch.nn.Parameter] = []  # apollo_params + decay_params + nodecay_params
    decay_param_names = _get_decay_parameter_names(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            if id(param) not in id_apollo_params:
                if name in decay_param_names:
                    decay_params.append(param)
                else:
                    nodecay_params.append(param)

    _, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

    if training_args.optim == "adamw_torch":
        optim_class = APOLLOAdamW
    else:
        raise NotImplementedError(f"Unknown optim: {training_args.optim}.")

    if finetuning_args.apollo_layerwise:
        logger.warning_rank0("The displayed gradient norm will be all zeros in layerwise APOLLO.")
        if training_args.gradient_accumulation_steps != 1:
            raise ValueError("Per-layer APOLLO does not support gradient accumulation.")

        optimizer_dict: dict[torch.Tensor, torch.optim.Optimizer] = {}
        for param in nodecay_params:
            param_groups = [dict(params=[param], weight_decay=0.0)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)
        for param in decay_params:
            param_groups = [dict(params=[param], weight_decay=training_args.weight_decay)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)
        for param in apollo_params:  # apollo params have weight decay
            param_groups = [dict(params=[param], weight_decay=training_args.weight_decay, **apollo_kwargs)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)

        def optimizer_hook(param: "torch.nn.Parameter"):
            if param.grad is not None:
                optimizer_dict[param].step()
                optimizer_dict[param].zero_grad()

        for param in trainable_params:
            param.register_post_accumulate_grad_hook(optimizer_hook)

        optimizer = DummyOptimizer(lr=training_args.learning_rate, optimizer_dict=optimizer_dict)
    else:
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0),
            dict(params=decay_params, weight_decay=training_args.weight_decay),
            dict(params=apollo_params, weight_decay=training_args.weight_decay, **apollo_kwargs),
        ]
        optimizer = optim_class(param_groups, **optim_kwargs)

    logger.info_rank0(f"Using APOLLO optimizer with args: {apollo_kwargs}.")
    return optimizer


def _create_loraplus_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    default_lr = training_args.learning_rate
    loraplus_lr = training_args.learning_rate * finetuning_args.loraplus_lr_ratio
    embedding_lr = finetuning_args.loraplus_lr_embedding

    decay_param_names = _get_decay_parameter_names(model)
    param_dict: dict[str, list[torch.nn.Parameter]] = {
        "lora_a": [],
        "lora_b": [],
        "lora_b_nodecay": [],
        "embedding": [],
    }
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_embedding_B" in name:
                param_dict["embedding"].append(param)
            elif "lora_B" in name or param.ndim == 1:
                if name in decay_param_names:
                    param_dict["lora_b"].append(param)
                else:
                    param_dict["lora_b_nodecay"].append(param)
            else:
                param_dict["lora_a"].append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=param_dict["lora_a"], lr=default_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b"], lr=loraplus_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b_nodecay"], lr=loraplus_lr, weight_decay=0.0),
        dict(params=param_dict["embedding"], lr=embedding_lr, weight_decay=training_args.weight_decay),
    ]
    optimizer = optim_class(param_groups, **optim_kwargs)
    logger.info_rank0(f"Using LoRA+ optimizer with loraplus lr ratio {finetuning_args.loraplus_lr_ratio:.2f}.")
    return optimizer


def _create_badam_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    decay_params, nodecay_params = [], []
    decay_param_names = _get_decay_parameter_names(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in decay_param_names:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=nodecay_params, weight_decay=0.0),
        dict(params=decay_params, weight_decay=training_args.weight_decay),
    ]

    if finetuning_args.badam_mode == "layer":
        from badam import BlockOptimizer  # type: ignore

        base_optimizer = optim_class(param_groups, **optim_kwargs)
        optimizer = BlockOptimizer(
            base_optimizer=base_optimizer,
            named_parameters_list=list(model.named_parameters()),
            block_prefix_list=None,
            switch_block_every=finetuning_args.badam_switch_interval,
            start_block=finetuning_args.badam_start_block,
            switch_mode=finetuning_args.badam_switch_mode,
            verbose=finetuning_args.badam_verbose,
            ds_zero3_enabled=is_deepspeed_zero3_enabled(),
        )
        logger.info_rank0(
            f"Using BAdam optimizer with layer-wise update, switch mode is {finetuning_args.badam_switch_mode}, "
            f"switch block every {finetuning_args.badam_switch_interval} steps, "
            f"default start block is {finetuning_args.badam_start_block}"
        )

    elif finetuning_args.badam_mode == "ratio":
        from badam import BlockOptimizerRatio  # type: ignore

        assert finetuning_args.badam_update_ratio > 1e-6
        optimizer = BlockOptimizerRatio(
            param_groups=param_groups,
            named_parameters_list=list(model.named_parameters()),
            update_ratio=finetuning_args.badam_update_ratio,
            mask_mode=finetuning_args.badam_mask_mode,
            verbose=finetuning_args.badam_verbose,
            include_embedding=False,
            **optim_kwargs,
        )
        logger.info_rank0(
            f"Using BAdam optimizer with ratio-based update, update ratio is {finetuning_args.badam_update_ratio}, "
            f"mask mode is {finetuning_args.badam_mask_mode}"
        )

    return optimizer


def _create_adam_mini_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
) -> "torch.optim.Optimizer":
    from adam_mini import Adam_mini  # type: ignore

    hidden_size = getattr(model.config, "hidden_size", None)
    num_q_head = getattr(model.config, "num_attention_heads", None)
    num_kv_head = getattr(model.config, "num_key_value_heads", None)

    optimizer = Adam_mini(
        named_parameters=model.named_parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        model_sharding=is_fsdp_enabled() or is_deepspeed_zero3_enabled(),
        dim=hidden_size,
        n_heads=num_q_head,
        n_kv_heads=num_kv_head,
    )
    logger.info_rank0("Using Adam-mini optimizer.")
    return optimizer


def _create_muon_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
) -> "torch.optim.Optimizer":
    from ..third_party.muon import Muon

    muon_params, adamw_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Use Muon for 2D parameters that aren't embeddings or heads
            if param.ndim == 2 and "embed" not in name and "lm_head" not in name:
                muon_params.append(param)
            else:
                adamw_params.append(param)

    optimizer = Muon(
        lr=training_args.learning_rate,
        wd=training_args.weight_decay,
        muon_params=muon_params,
        adamw_params=adamw_params,
        adamw_betas=(training_args.adam_beta1, training_args.adam_beta2),
        adamw_eps=training_args.adam_epsilon,
    )
    logger.info_rank0(
        f"Using Muon optimizer with {len(muon_params)} Muon params and {len(adamw_params)} AdamW params."
    )
    return optimizer


def create_custom_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> Optional["torch.optim.Optimizer"]:
    if finetuning_args.use_galore:
        return _create_galore_optimizer(model, training_args, finetuning_args)

    if finetuning_args.use_apollo:
        return _create_apollo_optimizer(model, training_args, finetuning_args)

    if finetuning_args.loraplus_lr_ratio is not None:
        return _create_loraplus_optimizer(model, training_args, finetuning_args)

    if finetuning_args.use_badam:
        return _create_badam_optimizer(model, training_args, finetuning_args)

    if finetuning_args.use_adam_mini:
        return _create_adam_mini_optimizer(model, training_args)

    if finetuning_args.use_muon:
        return _create_muon_optimizer(model, training_args)


def create_custom_scheduler(
    training_args: "TrainingArguments",
    num_training_steps: int,
    optimizer: Optional["torch.optim.Optimizer"] = None,
) -> None:
    if training_args.lr_scheduler_type == "warmup_stable_decay":
        num_warmup_steps = training_args.get_warmup_steps(num_training_steps)
        remaining_steps = num_training_steps - num_warmup_steps
        num_stable_steps = remaining_steps // 3  # use 1/3 for stable by default
        num_decay_steps = remaining_steps - num_stable_steps
        scheduler_kwargs = training_args.lr_scheduler_kwargs or {}
        default_kwargs = {
            "num_stable_steps": num_stable_steps,
            "num_decay_steps": num_decay_steps,
        }
        for key, value in default_kwargs.items():
            if key not in scheduler_kwargs:
                scheduler_kwargs[key] = value

        training_args.lr_scheduler_kwargs = scheduler_kwargs

    if optimizer is not None and isinstance(optimizer, DummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict: dict[torch.nn.Parameter, torch.optim.lr_scheduler.LRScheduler] = {}

        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer=optimizer_dict[param],
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
            )

        def scheduler_hook(param: "torch.nn.Parameter"):
            scheduler_dict[param].step()

        for param in optimizer_dict.keys():
            param.register_post_accumulate_grad_hook(scheduler_hook)


def get_batch_logps(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    label_pad_token_id: int = IGNORE_INDEX,
    ld_alpha: Optional[float] = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    r"""Compute the log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.

    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    valid_length = loss_mask.sum(-1)
    if ld_alpha is not None:
        num_examples = labels.shape[0] // 2
        chosen_lengths = valid_length[:num_examples]
        rejected_lengths = valid_length[num_examples:]
        min_lengths = torch.min(chosen_lengths, rejected_lengths)
        start_positions = torch.argmax(loss_mask.int(), dim=1)
        public_lengths = start_positions + torch.cat([min_lengths, min_lengths], dim=0)

        seq_len = labels.shape[-1]
        position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

        ld_mask = position_ids < public_lengths.unsqueeze(1)
        front_mask = (ld_mask * loss_mask).float()
        rear_mask = (~ld_mask * loss_mask).float()

        front_logps = (per_token_logps * front_mask).sum(-1)
        rear_logps = (per_token_logps * rear_mask).sum(-1)
        logps = front_logps + ld_alpha * rear_logps
    else:
        logps = (per_token_logps * loss_mask).sum(-1)

    return logps, valid_length


def dft_loss_func(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits")
    if logits is None:
        return outputs.get("loss", torch.tensor(0.0))

    logits = logits.float()
    vocab_size = logits.size(-1)
    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)

    loss = _dft_cross_entropy(logits, shift_labels, num_items_in_batch)
    return loss


def _dft_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    per_token_loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction="none")
    valid_mask = target != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=source.device, dtype=source.dtype)

    valid_losses = per_token_loss[valid_mask]

    with torch.no_grad():
        target_probs = torch.exp(-valid_losses)

    weighted_losses = valid_losses * target_probs

    if num_items_in_batch is not None:
        total_loss = weighted_losses.sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total_loss.device)
        loss = total_loss / num_items_in_batch
    else:
        loss = weighted_losses.mean()
    return loss


def nested_detach(
    tensors: Union["torch.Tensor", list["torch.Tensor"], tuple["torch.Tensor"], dict[str, "torch.Tensor"]],
    clone: bool = False,
):
    r"""Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t, clone=clone) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t, clone=clone) for k, t in tensors.items()})

    if isinstance(tensors, torch.Tensor):
        if clone:
            return tensors.detach().clone()
        else:
            return tensors.detach()
    else:
        return tensors


def get_swanlab_callback(finetuning_args: "FinetuningArguments") -> "TrainerCallback":
    r"""Get the callback for logging to SwanLab."""
    import swanlab  # type: ignore
    from swanlab.integration.transformers import SwanLabCallback  # type: ignore

    if finetuning_args.swanlab_api_key is not None:
        swanlab.login(api_key=finetuning_args.swanlab_api_key)

    if finetuning_args.swanlab_lark_webhook_url is not None:
        from swanlab.plugin.notification import LarkCallback  # type: ignore

        lark_callback = LarkCallback(
            webhook_url=finetuning_args.swanlab_lark_webhook_url,
            secret=finetuning_args.swanlab_lark_secret,
        )
        swanlab.register_callbacks([lark_callback])

    class SwanLabCallbackExtension(SwanLabCallback):
        def setup(self, args: "TrainingArguments", state: "TrainerState", model: "PreTrainedModel", **kwargs):
            if not state.is_world_process_zero:
                return

            super().setup(args, state, model, **kwargs)
            try:
                if hasattr(self, "_swanlab"):
                    swanlab_public_config = self._swanlab.get_run().public.json()
                else:  # swanlab <= 0.4.9
                    swanlab_public_config = self._experiment.get_run().public.json()
            except Exception:
                swanlab_public_config = {}

            with open(os.path.join(args.output_dir, SWANLAB_CONFIG), "w") as f:
                f.write(json.dumps(swanlab_public_config, indent=2))

    swanlab_callback = SwanLabCallbackExtension(
        project=finetuning_args.swanlab_project,
        workspace=finetuning_args.swanlab_workspace,
        experiment_name=finetuning_args.swanlab_run_name,
        mode=finetuning_args.swanlab_mode,
        config={"Framework": "🦙LlamaFactory"},
        logdir=finetuning_args.swanlab_logdir,
        tags=["🦙LlamaFactory"],
    )
    return swanlab_callback


def get_ray_trainer(
    training_function: Callable,
    train_loop_config: dict[str, Any],
    ray_args: "RayArguments",
) -> "TorchTrainer":
    if not ray_args.use_ray:
        raise ValueError("Ray was not enabled. Please set `USE_RAY=1` to enable ray.")

    if ray_args.ray_init_kwargs is not None:
        ray.init(**ray_args.ray_init_kwargs)

    if ray_args.ray_storage_filesystem is not None:
        # this means we are using s3/gcs
        storage_path = ray_args.ray_storage_path
    else:
        storage_path = Path(ray_args.ray_storage_path).absolute().as_posix()

    trainer = TorchTrainer(
        training_function,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(
            num_workers=ray_args.ray_num_workers,
            resources_per_worker=ray_args.resources_per_worker,
            placement_strategy=ray_args.placement_strategy,
            use_gpu=True,
        ),
        run_config=RunConfig(
            name=ray_args.ray_run_name,
            storage_filesystem=ray_args.ray_storage_filesystem,
            storage_path=storage_path,
        ),
    )
    return trainer
