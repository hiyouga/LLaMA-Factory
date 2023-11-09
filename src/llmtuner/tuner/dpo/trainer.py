import torch
from copy import deepcopy
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union
from transformers import BatchEncoding, Trainer
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model

from llmtuner.extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from trl import PreTrainedModelWrapper


class CustomDPOTrainer(DPOTrainer):

    def __init__(
        self,
        beta: float,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]] = None,
        disable_dropout: Optional[bool] = True,
        loss_type: Optional[Literal["sigmoid", "hinge"]] = "sigmoid",
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
        self.loss_type = loss_type
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: "PreTrainedModelWrapper"):
        # adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0

        # Lazy load
        import deepspeed # type: ignore
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def concatenated_forward(
        self,
        model: Optional[torch.nn.Module] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None
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
