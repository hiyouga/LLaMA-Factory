from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
from transformers import BatchEncoding, Trainer
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class CustomDPOTrainer(DPOTrainer):
    """
    Custom trainer for Differentiable Policy Optimization (DPO).

    This class extends DPOTrainer and provides additional functionalities for DPO training.

    Parameters
    ----------
    beta : float
        Coefficient for controlling the importance of the DPO loss term.

    loss_type : Literal["sigmoid", "hinge", "ipo", "kto"]
        Type of loss function to use.

    ftx_gamma : float
        Coefficient for controlling the importance of the supervised cross-entropy loss term.

    model : Union["PreTrainedModel", torch.nn.Module]
        The main model for training.

    ref_model : Optional[Union["PreTrainedModel", torch.nn.Module]], optional
        Reference model for comparison during training, by default None.

    disable_dropout : Optional[bool], optional
        Flag to disable dropout during initialization, by default True.

    **kwargs : Any
        Additional keyword arguments.

    Raises
    ------
    AttributeError
        If `transformers` needs to be updated.

    ValueError
        If the task is unknown.

    Attributes
    ----------
    use_dpo_data_collator : bool
        Flag indicating whether to use DPO data collator.

    generate_during_eval : bool
        Flag indicating whether to generate during evaluation.

    label_pad_token_id : int
        Token ID for padding labels.

    padding_value : int
        Value for padding.

    is_encoder_decoder : bool
        Flag indicating if the model is an encoder-decoder architecture.
    
    precompute_ref_log_probs : bool
        Flag indicating whether to precompute reference log probabilities.
    
    _precomputed_train_ref_log_probs : bool
        Flag indicating whether reference log probabilities are precomputed for training.
    
    _precomputed_eval_ref_log_probs : bool
        Flag indicating whether reference log probabilities are precomputed for evaluation.
    
    _peft_has_been_casted_to_bf16 : bool
        Flag indicating if PEFT has been casted to BF16.
    
    ref_model : Optional[Union["PreTrainedModel", torch.nn.Module]]
        Reference model for comparison during training.
    
    beta : float
        Coefficient for controlling the importance of the DPO loss term.
    
    label_smoothing : int
        Coefficient for controlling label smoothing.
    
    loss_type : Literal["sigmoid", "hinge", "ipo", "kto"]
        Type of loss function to use.
    
    ftx_gamma : float
        Coefficient for controlling the importance of the supervised cross-entropy loss term.
    
    _stored_metrics : defaultdict
        Dictionary to store metrics.

    Methods
    -------
    sft_loss(chosen_logits: torch.FloatTensor, chosen_labels: torch.LongTensor) -> torch.Tensor:
        Computes supervised cross-entropy loss of given labels under the given logits.

    concatenated_forward(model: "PreTrainedModel", batch: Dict[str, torch.Tensor]) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        Computes the forward pass for concatenated inputs.

    get_batch_loss_metrics(model: "PreTrainedModel", batch: Dict[str, torch.Tensor], train_eval: Optional[Literal["train", "eval"]] = "train") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
    """
    def __init__(
        self,
        beta: float,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto"],
        ftx_gamma: float,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]] = None,
        disable_dropout: Optional[bool] = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

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
        self.beta = beta
        self.label_smoothing = 0
        self.loss_type = loss_type
        self.ftx_gamma = ftx_gamma
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

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

    def sft_loss(self, chosen_logits: torch.FloatTensor, chosen_labels: torch.LongTensor) -> torch.Tensor:
        """
        Computes supervised cross-entropy loss of given labels under the given logits.

        Parameters
        ----------
        chosen_logits : torch.FloatTensor
            Logits for chosen labels.

        chosen_labels : torch.LongTensor
            Chosen labels.

        Returns
        -------
        torch.Tensor
            Tensor containing the cross-entropy loss of each sample.
        """
        all_logps = self.get_batch_logps(chosen_logits, chosen_labels, average_log_prob=True)
        return -all_logps

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Computes the forward pass for concatenated inputs.

        Parameters
        ----------
        model : "PreTrainedModel"
            The model.

        batch : Dict[str, torch.Tensor]
            Dictionary containing the batched data.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
            Tuple containing chosen and rejected log probabilities and logits.
        """
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()})  # avoid error

        all_logits = model(
            input_ids=batch_copied["input_ids"], attention_mask=batch_copied["attention_mask"], return_dict=True
        ).logits.to(torch.float32)

        all_logps = self.get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
        )
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, torch.Tensor],
        train_eval: Optional[Literal["train", "eval"]] = "train",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.

        Parameters
        ----------
        model : "PreTrainedModel"
            The model.

        batch : Dict[str, torch.Tensor]
            Dictionary containing the batched data.

        train_eval : Optional[Literal["train", "eval"]], optional
            Flag indicating whether it's train or evaluation mode, by default "train".

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Tuple containing loss and metrics.
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
                ) = self.concatenated_forward(ref_model, batch)

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
