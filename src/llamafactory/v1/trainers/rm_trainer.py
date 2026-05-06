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
import torch.nn.functional as F

from ..accelerator.interface import DistributedInterface
from ..config import InputArgument, TrainingArguments, get_args
from ..config.arg_utils import ModelClass
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..utils import logging
from ..utils.types import BatchInput, HFModel, Tensor


logger = logging.get_logger(__name__)


def _prepare_4d_attention_mask(attention_mask_with_indices: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Expand 2D attention mask with document indices to 4D block-diagonal causal mask.

    Input: (batch_size, seq_len) with values like [1,1,2,2,0] (0=padding, different ints=different docs)
    Output: (batch_size, 1, seq_len, seq_len) with 0.0 for attend, min_dtype for mask.
    """
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype, device=attention_mask_with_indices.device)

    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)
    tril_mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=attention_mask_with_indices.device)
    )
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d


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

        super().__init__(args, model, renderer, train_dataset, callbacks)

    def _get_score_module(self):
        """Get the score head module from the SeqCls model for hook registration."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        score = getattr(model, "score", None)
        if score is None or getattr(model, "model", None) is None:
            raise ValueError(
                "RM training requires a model loaded with AutoModelForSequenceClassification "
                "(model_class='cls_seq'). The model must have `.model` and `.score` attributes."
            )
        return score

    def compute_loss(self, batch: BatchInput) -> Tensor:
        if "attention_mask" not in batch:
            raise ValueError("RM training requires attention_mask in batch.")

        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)

        position_ids = None
        if "position_ids" in batch and isinstance(batch["position_ids"], torch.Tensor):
            position_ids = batch["position_ids"].to(self.device, non_blocking=True)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is None:
            raise ValueError(
                "RM training requires pair data with token_type_ids. "
                "Ensure the dataset has chosen_messages/rejected_messages."
            )
        token_type_ids = token_type_ids.to(self.device, non_blocking=True)

        attn_impl = getattr(self.model.config, "_attn_implementation", "eager")
        if attn_impl == "flash_attention_2":
            model_attention_mask = attention_mask
        else:
            model_attention_mask = _prepare_4d_attention_mask(attention_mask, dtype=torch.float32)

        # Call through the full FSDP-wrapped model instead of sub-modules directly.
        # Under FSDP2, calling sub-modules bypasses the pre-forward hooks that handle
        # DTensor unshard / input casting, causing mixed Tensor/DTensor errors.
        # A forward hook on the score layer captures per-position scores before pooling.
        score_module = self._get_score_module()
        captured_scores: dict[str, torch.Tensor] = {}

        def _capture_hook(_module, _input, output):
            captured_scores["value"] = output

        hook_handle = score_module.register_forward_hook(_capture_hook)
        try:
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=model_attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
        finally:
            hook_handle.remove()

        rewards = captured_scores["value"].float().squeeze(-1)

        # FSDP2 registers its backward hooks (which unshard parameters) on the module's
        # output tensors.  Since loss is computed from the captured intermediate tensor
        # rather than model_output, we must keep model_output in the autograd graph so
        # those backward hooks still fire and parameters are unsharded during backward.
        _fsdp_bwd_anchor = model_output.logits.sum() * 0.0

        chosen_mask = (token_type_ids == 1)
        rejected_mask = (token_type_ids == 2)

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
        return -F.logsigmoid(chosen_scores - rejected_scores).mean() + _fsdp_bwd_anchor


def run_rm(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    model_args.model_class = ModelClass.CLS_SEQ
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
    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    run_rm()
