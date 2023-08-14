import torch
from collections import defaultdict
from peft import PeftModel
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from transformers import BatchEncoding, Trainer
from trl import DPOTrainer

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.tuner.core.trainer import PeftModelMixin

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from llmtuner.hparams import FinetuningArguments, GeneratingArguments


class DPOPeftTrainer(PeftModelMixin, DPOTrainer):

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]] = None,
        **kwargs
    ):
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        self.ref_model = ref_model
        self.use_dpo_data_collator = True # hack to avoid warning
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.beta = finetuning_args.dpo_beta
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        Trainer.__init__(self, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def concatenated_forward(
        self,
        model: Optional[torch.nn.Module] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()}) # avoid error
        unwrapped_model: "PreTrainedModel" = self.accelerator.unwrap_model(self.model)

        if not torch.is_grad_enabled():
            unwrapped_model.gradient_checkpointing_disable()

        if model is None and isinstance(unwrapped_model, PeftModel): # peft model has no ref_model
            with unwrapped_model.disable_adapter():
                all_logits = self.model(
                    input_ids=batch_copied["input_ids"],
                    attention_mask=batch_copied["attention_mask"],
                    return_dict=True
                ).logits.to(torch.float32)
        else:
            all_logits = model(
                input_ids=batch_copied["input_ids"],
                attention_mask=batch_copied["attention_mask"],
                return_dict=True
            ).logits.to(torch.float32)

        if not torch.is_grad_enabled():
            unwrapped_model.gradient_checkpointing_enable()

        all_logps = self._get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False
        )
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits
