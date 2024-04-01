from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX
from ..utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments


class CustomORPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", "torch.nn.Module"],
        finetuning_args: "FinetuningArguments",
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)

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

        self.beta = finetuning_args.orpo_beta
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        Trainer.__init__(self, model=model, **kwargs)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        odds_ratio_loss = -F.logsigmoid(log_odds)
        return odds_ratio_loss

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the average log probabilities of the labels under the given logits.
        """
        all_logits: "torch.Tensor" = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_dict=True, use_cache=False
        ).logits.to(torch.float32)

        all_logps = self.get_batch_logps(
            logits=all_logits,
            labels=batch["labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the ORPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        chosen_logps, rejected_logps, chosen_logits, rejected_logits = self.concatenated_forward(model, batch)
        sft_loss = -chosen_logps
        odds_ratio_loss = self.odds_ratio_loss(chosen_logps, rejected_logps)
        batch_loss = (sft_loss + self.beta * odds_ratio_loss).mean()

        chosen_rewards = self.beta * chosen_logps.detach()
        rejected_rewards = self.beta * rejected_logps.detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.cpu().mean()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.cpu().mean()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.cpu().mean()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics["{}logps/rejected".format(prefix)] = rejected_logps.detach().cpu().mean()
        metrics["{}logps/chosen".format(prefix)] = chosen_logps.detach().cpu().mean()
        metrics["{}logits/rejected".format(prefix)] = rejected_logits.detach().cpu().mean()
        metrics["{}logits/chosen".format(prefix)] = chosen_logits.detach().cpu().mean()
        metrics["{}sft_loss".format(prefix)] = sft_loss.detach().cpu().mean()
        metrics["{}odds_ratio_loss".format(prefix)] = odds_ratio_loss.detach().cpu().mean()

        return batch_loss, metrics
