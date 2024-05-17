from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from transformers import Trainer
from trl import KTOTrainer
from trl.trainer.utils import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX
from ..utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments


class CustomKTOTrainer(KTOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
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

        # KTO parameter
        self.beta = finetuning_args.kto_beta
        self.ftx_gamma = finetuning_args.kto_ftx
        self.desirable_weight = finetuning_args.kto_desirable_weight
        self.undesirable_weight = finetuning_args.kto_undesirable_weight


        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def sft_loss(self, chosen_logits: "torch.FloatTensor", chosen_labels: "torch.LongTensor") -> "torch.Tensor":
        r"""
        Computes supervised cross-entropy loss of given labels under the given logits.
        Returns:
            A tensor of shape (batch_size,) containing the cross-entropy loss of each samples.
        """
        all_logps = self.get_batch_logps(chosen_logits, chosen_labels, average_log_prob=True)
        return -all_logps.nanmean()


    def forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        with torch.no_grad():
            KL_logits = model(
                batch["KL_completion_input_ids"],
                attention_mask=batch["KL_completion_attention_mask"],
            ).logits

        completion_logits = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            batch["labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        KL_logps = self.get_batch_logps(
            KL_logits,
            batch["kl_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        if completion_logps.shape[0] != len(batch["tag"]):
            raise ValueError(
                "There is a mismatch between the number of examples in this batch and the number of "
                "examples for which an output sequence was predicted."
            )
        chosen_idx = [i for i in range(completion_logps.shape[0]) if batch["tag"][i]]
        rejected_idx = [i for i in range(completion_logps.shape[0]) if not batch["tag"][i]]

        chosen_logps = completion_logps[chosen_idx, ...]
        rejected_logps = completion_logps[rejected_idx, ...]

        chosen_logits = completion_logits[chosen_idx, ...]
        rejected_logits = completion_logits[rejected_idx, ...]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps)


    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_KL_logps,
        ) = self.forward(model, batch)

        with torch.no_grad():
            if self.ref_model is None:
                ref_model = self.model
                ref_context = self.accelerator.unwrap_model(self.model).disable_adapter()
            else:
                ref_model = self.ref_model
                ref_context = nullcontext()
            with ref_context:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    reference_KL_logps,
                ) = self.forward(ref_model, batch)

        losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_KL_logps,
        )
        losses = losses.nanmean()
        if self.ftx_gamma > 1e-6 and len(batch["labels"][batch['tag']])>0:
            losses += self.ftx_gamma * self.sft_loss(policy_chosen_logits, batch["labels"][batch['tag']])


        num_chosen = torch.Tensor([len(chosen_rewards)]).to(self.accelerator.device)
        num_rejected = torch.Tensor([len(rejected_rewards)]).to(self.accelerator.device)

        all_num_chosen = self.accelerator.gather(num_chosen).sum().item()
        all_num_rejected = self.accelerator.gather(num_rejected).sum().item()

        if all_num_chosen > 0:
            metrics["rewards/chosen_sum"] = self.accelerator.gather(chosen_rewards.nansum()).nansum().item()
            metrics["logps/chosen_sum"] = self.accelerator.gather(policy_chosen_logps.nansum()).nansum().item()
            metrics["count/chosen"] = all_num_chosen

        if all_num_rejected > 0:
            metrics["rewards/rejected_sum"] = self.accelerator.gather(rejected_rewards.nansum()).nansum().item()
            metrics["logps/rejected_sum"] = self.accelerator.gather(policy_rejected_logps.nansum()).nansum().item()
            metrics["count/rejected"] = all_num_rejected

        metrics["kl"] = kl.item()

        return losses, metrics