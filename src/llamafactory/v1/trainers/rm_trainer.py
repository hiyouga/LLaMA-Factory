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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..accelerator.interface import DistributedInterface
from ..config import InputArgument, TrainingArguments, get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..core.utils.reward_head import load_reward_head, save_reward_head
from ..utils import logging
from ..utils.types import BatchInput, HFModel, Tensor


logger = logging.get_logger(__name__)


def _get_hidden_size(model: HFModel) -> int:
    config = model.config
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError("Cannot infer hidden size from model config for RM training.")


def _attach_reward_head(model: HFModel) -> None:
    if hasattr(model, "reward_head"):
        return

    hidden_size = _get_hidden_size(model)
    reward_head = nn.Linear(hidden_size, 1, bias=False)
    nn.init.normal_(reward_head.weight, mean=0.0, std=0.02)
    try:
        ref_param = next(model.parameters())
        reward_head = reward_head.to(device=ref_param.device, dtype=ref_param.dtype)
    except StopIteration:
        # Fallback for models without parameters (unexpected), keep default init device/dtype.
        pass
    model.add_module("reward_head", reward_head)


def _validate_rm_dataset_format(train_dataset: DataEngine, dataset_path: str) -> None:
    """Validate RM dataset format early for clearer error messages."""
    if len(train_dataset) == 0:
        raise ValueError(f"RM training dataset is empty: {dataset_path}")

    sample = train_dataset[0]
    if "chosen_messages" in sample and "rejected_messages" in sample:
        return

    dataset_name = sample.get("_dataset_name", "unknown")
    sample_keys = sorted(sample.keys())
    raise ValueError(
        "RM training requires pair-format samples containing chosen/rejected responses. "
        f"First sample from dataset '{dataset_name}' has keys: {sample_keys}. "
        "Please use pair data (e.g. a dataset with chosen_messages/rejected_messages, "
        "or set converter='pair' for raw chosen/rejected fields)."
    )


class RMTrainer(BaseTrainer):
    def __init__(
        self,
        args: TrainingArguments,
        model: HFModel,
        renderer,
        train_dataset,
        callbacks=None,
    ) -> None:
        cp_size = args.dist_config.get("cp_size", 1) if args.dist_config is not None else 1
        if cp_size > 1:
            raise NotImplementedError("RM trainer currently only supports cp_size == 1.")

        _attach_reward_head(model)
        super().__init__(args, model, renderer, train_dataset, callbacks)

    def save_model(self) -> None:
        super().save_model()
        if DistributedInterface().get_rank() == 0 and save_reward_head(self.model, self.args.output_dir):
            logger.info_rank0(f"Reward head model saved at: {self.args.output_dir}")

    def compute_loss(self, batch: BatchInput) -> Tensor:
        if "token_type_ids" not in batch:
            raise ValueError("RM training requires pair data with token_type_ids from converter='pair'.")
        if "attention_mask" not in batch:
            raise ValueError("RM training requires attention_mask in batch.")

        model_inputs: dict[str, Tensor] = {}
        for key in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                model_inputs[key] = batch[key].to(self.device, non_blocking=True)

        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        reward_head_dtype = self.model.reward_head.weight.dtype
        if hidden_states.dtype != reward_head_dtype:
            hidden_states = hidden_states.to(dtype=reward_head_dtype)
        rewards = self.model.reward_head(hidden_states).squeeze(-1)

        token_type_ids = batch["token_type_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True).bool()
        chosen_mask = (token_type_ids == 1) & attention_mask
        rejected_mask = (token_type_ids == 2) & attention_mask

        valid_pair_mask = chosen_mask.any(dim=-1) & rejected_mask.any(dim=-1)
        if not torch.any(valid_pair_mask):
            raise ValueError(
                "No valid RM pairs found in this micro-batch. "
                "This is usually caused by cutoff_len being too small and truncating chosen/rejected tokens."
            )

        rewards = rewards[valid_pair_mask]
        chosen_mask = chosen_mask[valid_pair_mask]
        rejected_mask = rejected_mask[valid_pair_mask]

        seq_len = rewards.size(-1)
        position_index = torch.arange(seq_len, device=self.device).unsqueeze(0)
        chosen_last_idx = (position_index * chosen_mask.long()).max(dim=-1).values
        rejected_last_idx = (position_index * rejected_mask.long()).max(dim=-1).values

        chosen_scores = rewards.gather(dim=1, index=chosen_last_idx.unsqueeze(-1)).squeeze(-1)
        rejected_scores = rewards.gather(dim=1, index=rejected_last_idx.unsqueeze(-1)).squeeze(-1)
        return -F.logsigmoid(chosen_scores.float() - rejected_scores.float()).mean()


def run_rm(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    DistributedInterface(training_args.dist_config)
    train_dataset = DataEngine(data_args.train_dataset)
    _validate_rm_dataset_format(train_dataset, data_args.train_dataset)
    model_engine = ModelEngine(model_args, is_train=True)
    trainer = RMTrainer(
        args=training_args,
        model=model_engine.model,
        renderer=model_engine.renderer,
        train_dataset=train_dataset,
    )
    if os.path.isdir(model_args.model) and load_reward_head(trainer.model, model_args.model, trainer.device):
        logger.info_rank0(f"Loaded reward head weights from: {model_args.model}")
    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    run_rm()
