import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger
from llmtuner.tuner.core.trainer import PeftTrainer

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput


logger = get_logger(__name__)


class Seq2SeqPeftTrainer(PeftTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
        if prompt_len > label_len:
            inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
        if label_len > prompt_len:
            inputs["input_ids"] = self._pad_tensors_to_target_len(inputs["input_ids"], inputs["labels"])
            if "attention_mask" in inputs:
                inputs["attention_mask"] = self._pad_tensors_to_target_len(
                    inputs["attention_mask"], inputs["labels"], pad_token_id=0
                )
            if "position_ids" in inputs:
                inputs["position_ids"] = self._pad_tensors_to_target_len(
                    inputs["position_ids"], inputs["labels"], pad_token_id=0
                )

        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None:
            generated_tokens[:, :max(prompt_len, label_len)] = (
                self.tokenizer.pad_token_id * torch.ones_like(generated_tokens[:, :max(prompt_len, label_len)])
            )

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.

        Should only be called when predict_with_generate=True.
        """
        if pad_token_id is None:
            if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
                assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
                pad_token_id = self.tokenizer.pad_token_id
            else:
                raise ValueError("PAD token is required.")

        padded_tensor = pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        return padded_tensor.contiguous() # in contiguous memory

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))
