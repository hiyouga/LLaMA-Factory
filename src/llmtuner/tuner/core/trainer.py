import os
import torch
from typing import TYPE_CHECKING, Dict, Optional

from transformers import Seq2SeqTrainer
from transformers.trainer import TRAINING_ARGS_NAME, WEIGHTS_NAME
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from peft import PeftModel
from trl import PreTrainedModelWrapper

from llmtuner.extras.constants import FINETUNING_ARGS_NAME, VALUE_HEAD_FILE_NAME
from llmtuner.extras.logging import get_logger
from llmtuner.extras.save_and_load import get_state_dict, load_trainable_params

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments, TrainerState
    from llmtuner.hparams import FinetuningArguments


logger = get_logger(__name__)


class PeftModelMixin:
    r"""
    Patches the save and load methods in Hugging Face Trainer for PeftModel and ModelWithValueHead.
    """

    def __init__(self) -> None: # for type checking
        self.model: PreTrainedModel = None
        self.tokenizer: "PreTrainedTokenizer" = None
        self.args: "Seq2SeqTrainingArguments" = None
        self.finetuning_args: "FinetuningArguments" = None
        self.state: "TrainerState" = None
        raise AssertionError("Mixin should not be initialized.")

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoint.

        This function will only be executed at the process zero.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model = unwrap_model(self.model)
        if isinstance(model, PreTrainedModelWrapper):
            # Custom state dict: https://github.com/lvwerra/trl/blob/v0.4.7/trl/models/modeling_value_head.py#L200
            model_state_dict = state_dict or model.state_dict()
            v_head_state_dict = {
                name.replace("v_head.", ""): model_state_dict[name].cpu().clone().detach()
                for name in model_state_dict.keys() if name.startswith("v_head.")
            }

            torch.save(v_head_state_dict, os.path.join(output_dir, VALUE_HEAD_FILE_NAME))
            model = model.pretrained_model

        state_dict = state_dict or get_state_dict(model)
        if isinstance(model, (PeftModel, PreTrainedModel)):
            model.config.use_cache = True
            model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
            model.config.use_cache = False
        else:
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.finetuning_args.finetuning_type == "full" and self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(output_dir)
            except:
                logger.warning("Cannot save tokenizer, copy the files manually.")

        with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
            f.write(self.args.to_json_string() + "\n")

        self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))

    def _load_best_model(self):
        r"""
        Loads trainable parameters from model checkpoint.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        model = unwrap_model(self.model)

        if isinstance(model, PreTrainedModelWrapper):
            model.v_head.load_state_dict(torch.load(
                os.path.join(self.state.best_model_checkpoint, VALUE_HEAD_FILE_NAME), map_location="cpu"
            ))
            model = model.pretrained_model

        if isinstance(model, PeftModel):
            model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)
        else: # freeze/full-tuning
            load_trainable_params(model, self.state.best_model_checkpoint)


class PeftTrainer(PeftModelMixin, Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
    """

    def __init__(self, finetuning_args: "FinetuningArguments", **kwargs):
        Seq2SeqTrainer.__init__(self, **kwargs)
        self.finetuning_args = finetuning_args
