import os
import json
import torch
import numpy as np
import torch.nn as nn
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import Seq2SeqTrainer, PreTrainedModel, Trainer
from peft import PeftModel

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(self, model: Union["PreTrainedModel", nn.Module] = None, neftune_noise_alpha: Optional[float] = 0, **kwargs):
        super().__init__(model, **kwargs)
        self.neftune_noise_alpha = neftune_noise_alpha
        self._neftune_activated = False

        if self.neftune_noise_alpha:
            self._activate_neftune(model)

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
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            assert self.tokenizer.pad_token_id is not None, "Pad token is required."
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
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :max(prompt_len, label_len)] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
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


    @wraps(Trainer.train)
    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return output

    def _toggle_neftune(self, model, activate=True):
        """Toggle NEFTune optimization for a model (i.e. activate or deactivate).
        This optimization based on this paper: https://arxiv.org/abs/2310.05914

        Parameters:
        model : PreTrainedModel or PeftModel
            The model to toggle the noise for.
        activate : bool, optional (default=True)
            Whether to activate the noise or not.
        """
        if activate == self._neftune_activated:
            return

        self._neftune_activated = activate

        embeddings = (model.get_input_embeddings() if isinstance(model, PreTrainedModel) 
                    else model.base_model.get_input_embeddings() if isinstance(model, PeftModel) 
                    else None)

        if embeddings:
            if activate:
                embeddings.neftune_noise_alpha = self.neftune_noise_alpha
                embeddings._trl_old_forward = embeddings.forward
                neftune_method = _neftune_forward_function.__get__(embeddings, embeddings.__class__)
                setattr(embeddings, "forward", neftune_method)
                logger.info("NEFTune activated with alpha: ", self.neftune_noise_alpha)
            elif hasattr(embeddings, "_trl_old_forward"):
                embeddings.forward = embeddings._trl_old_forward
                del embeddings._trl_old_forward
                del embeddings.neftune_noise_alpha
                logger.info("NEFTune deactivated")

    _activate_neftune = lambda self, model: self._toggle_neftune(model, activate=True)
    _deactivate_neftune = lambda self, model: self._toggle_neftune(model, activate=False)


def _neftune_forward_function(self, input: torch.Tensor) -> torch.Tensor:
    """
    This code is adapted from the original source code that can be found here: https://github.com/neelsjain/NEFTune
    """
    embeddings = torch.nn.functional.embedding(
        input,
        self.weight,
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse)

    if self.training:
        dims = torch.tensor(embeddings.size(1) * embeddings.size(2))
        mag_norm = self.neftune_noise_alpha / torch.sqrt(dims)
        embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)

    return embeddings
