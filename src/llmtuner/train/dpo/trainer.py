import torch
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union
from transformers import BatchEncoding, Trainer
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model

from llmtuner.extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class CustomDPOTrainer(DPOTrainer):

    def __init__(
        self,
        beta: float,
        loss_type: Literal["sigmoid", "hinge"],
        ftx_gamma: float,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]] = None,
        disable_dropout: Optional[bool] = True,
        **kwargs
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.ref_model = ref_model
        self.use_dpo_data_collator = True # hack to avoid warning
        self.generate_during_eval = False # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.beta = beta
        self.label_smoothing = 0
        self.ftx_gamma = ftx_gamma
        self.loss_type = loss_type
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False)
                    or getattr(ref_model, "is_loaded_in_4bit", False)
                ): # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def sft_loss(
        self,
        chosen_logits: torch.FloatTensor,
        chosen_labels: torch.LongTensor
    ) -> torch.Tensor:
        r"""
        Computes supervised cross-entropy loss of given labels under the given logits.

        Returns:
            A tensor of shape (batch_size,) containing the cross-entropy loss of each samples.
        """
        all_logps = self._get_batch_logps(
            chosen_logits,
            chosen_labels,
            average_log_prob=True
        )
        return -all_logps

    def concatenated_forward(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()}) # avoid error

        all_logits = model(
            input_ids=batch_copied["input_ids"],
            attention_mask=batch_copied["attention_mask"],
            return_dict=True
        ).logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False
        )
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def get_batch_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, torch.Tensor],
        train_eval: Optional[Literal["train", "eval"]] = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        if self.ftx_gamma > 1e-6:
            batch_size = batch["input_ids"].size(0) // 2
            chosen_labels, _ = batch["labels"].split(batch_size, dim=0)
            losses += self.ftx_gamma * self.sft_loss(policy_chosen_logits, chosen_labels)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics
