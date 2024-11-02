# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/kto_trainer.py
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

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
from transformers import Trainer
from trl import KTOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps


if TYPE_CHECKING:
    import torch.utils.data
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomKTOTrainer(KTOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # kto hyperparams
        self.beta = finetuning_args.pref_beta
        self.desirable_weight = finetuning_args.kto_chosen_weight
        self.undesirable_weight = finetuning_args.kto_rejected_weight
        self.ftx_gamma = finetuning_args.pref_ftx

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

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
        r"""
        Replaces the sequential sampler of KTO Trainer created by trl with the random sampler.
        """
        return Trainer._get_train_sampler(self)

    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        r"""
        Replaces the method of KTO Trainer with the one of the standard Trainer.
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    @override
    def forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"], prefix: Literal["", "kl_"] = ""
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Runs forward pass and computes the log probabilities.
        """
        batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error
        model_inputs = {
            "input_ids": batch[f"{prefix}input_ids"],
            "attention_mask": batch[f"{prefix}attention_mask"],
        }
        if f"{prefix}token_type_ids" in batch:
            model_inputs["token_type_ids"] = batch[f"{prefix}token_type_ids"]

        if "pixel_values" in batch:
            model_inputs["pixel_values"] = batch["pixel_values"]

        if "image_grid_thw" in batch:
            model_inputs["image_grid_thw"] = batch["image_grid_thw"]

        logits = model(**model_inputs, return_dict=True, use_cache=False).logits.to(torch.float32)
        logps, valid_length = get_batch_logps(logits=logits, labels=batch[f"{prefix}labels"])
        return logits, logps, logps / valid_length

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        target_logits, target_logps, target_logps_avg = self.forward(model, batch)
        with torch.no_grad():
            _, kl_logps, _ = self.forward(model, batch, prefix="kl_")

        if len(target_logps) != len(batch["kto_tags"]):
            raise ValueError("Mismatched shape of inputs and labels.")

        chosen_logits = target_logits[batch["kto_tags"]]
        chosen_logps = target_logps[batch["kto_tags"]]
        rejected_logits = target_logits[~batch["kto_tags"]]
        rejected_logps = target_logps[~batch["kto_tags"]]
        chosen_logps_avg = target_logps_avg[batch["kto_tags"]]
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, kl_logps, chosen_logps_avg

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes log probabilities of the reference model.
        """
        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, _, _, reference_kl_logps, _ = self.concatenated_forward(
                ref_model, batch
            )

        return reference_chosen_logps, reference_rejected_logps, reference_kl_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_kl_logps,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)
        reference_chosen_logps, reference_rejected_logps, reference_kl_logps = self.compute_reference_log_probs(
            model, batch
        )
        losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_kl_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_kl_logps,
        )
        losses = losses.nanmean()

        if self.ftx_gamma > 1e-6 and len(policy_chosen_logps) > 0:  # remember to rescale
            sft_loss = -policy_chosen_logps_avg
            losses += self.ftx_gamma * sft_loss.nanmean() / len(policy_chosen_logps) * len(batch["labels"])

        num_chosen = len(chosen_rewards)
        num_rejected = len(rejected_rewards)
        if num_chosen > 0:
            metrics["rewards/chosen_sum"] = chosen_rewards.nansum().item()
            metrics["logps/chosen_sum"] = policy_chosen_logps.nansum().item()
            metrics["logits/chosen_sum"] = policy_chosen_logits.nansum().item()
            metrics["count/chosen"] = float(num_chosen)

        if num_rejected > 0:
            metrics["rewards/rejected_sum"] = rejected_rewards.nansum().item()
            metrics["logps/rejected_sum"] = policy_rejected_logps.nansum().item()
            metrics["logits/rejected_sum"] = policy_rejected_logits.nansum().item()
            metrics["count/rejected"] = float(num_rejected)

        metrics["kl"] = kl.item()
        return losses, metrics

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        loss = super().compute_loss(model, inputs, return_outputs)
        if is_transformers_version_equal_to_4_46() and kwargs.pop("num_items_in_batch", False):
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def log(self, logs: Dict[str, float]) -> None:
        r"""
        Log `logs` on the various objects watching training, including stored metrics.
        """
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        prefix = "eval_" if train_eval == "eval" else ""
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).sum().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 9:  # pad to for all reduce
            for i in range(9 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "sum").tolist()
        metric_dict: Dict[str, float] = dict(zip(key_list, metric_list))
        for split in ["chosen", "rejected"]:  # accumulate average metrics from sums and lengths
            if f"count/{split}" in metric_dict:
                for key in ("rewards", "logps", "logits"):
                    logs[f"{prefix}{key}/{split}"] = metric_dict[f"{key}/{split}_sum"] / metric_dict[f"count/{split}"]
                    del metric_dict[f"{key}/{split}_sum"]
                del metric_dict[f"count/{split}"]

        if f"{prefix}rewards/chosen" in logs and f"{prefix}rewards/rejected" in logs:  # calculate reward margin
            logs[f"{prefix}rewards/margins"] = logs[f"{prefix}rewards/chosen"] - logs[f"{prefix}rewards/rejected"]

        for key, metric in metric_dict.items():  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs)
