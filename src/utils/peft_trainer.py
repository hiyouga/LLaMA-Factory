import os
import json
import time
import torch
from typing import Dict, Optional
from datetime import timedelta

from transformers import (
    Seq2SeqTrainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments
)

from transformers.trainer import TRAINING_ARGS_NAME
from transformers.modeling_utils import unwrap_model

from .config import FinetuningArguments

from .other import (
    get_logger,
    get_state_dict,
    load_trainable_params,
    load_valuehead_params,
    FINETUNING_ARGS_NAME,
    VALUE_HEAD_FILE_NAME
)


logger = get_logger(__name__)


class LogCallback(TrainerCallback):
    r"""
    TrainerCallback includes the state function during training, for more details refer to the TrainerCallback class.
    The on_log function primarily collects process parameters during training, such as training loss, learning rate,
    and training epochs, as well as progress parameters like the current percentage progress and estimated remaining
    time. Every time a log is triggered, a new record is appended to the file "messages.log" for dynamic visualization
    purposes.
    """

    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        r"""
        Event called after logging the last logs.
        """
        if "loss" not in state.log_history[-1]:
            return
        cur_time = time.time()
        cur_steps = state.log_history[-1].get("step")
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_steps = state.max_steps - cur_steps
        remaining_time = remaining_steps * avg_time_per_step
        log_dict = {
            "current_steps": cur_steps,
            "total_steps": state.max_steps,
            "loss": state.log_history[-1].get("loss", None),
            "reward": state.log_history[-1].get("reward", None),
            "learning_rate": state.log_history[-1].get("learning_rate", None),
            "epoch": state.log_history[-1].get("epoch", None),
            "percentage": round(cur_steps / state.max_steps * 100, 2) if state.max_steps != 0 else 100,
            "elapsed_time": str(timedelta(seconds=int(elapsed_time))),
            "remaining_time": str(timedelta(seconds=int(remaining_time)))
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "trainer_log.jsonl"), "a") as f:
            f.write(json.dumps(log_dict) + "\n")


class PeftTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
    """

    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if self.is_world_process_zero() and os.path.exists(os.path.join(self.args.output_dir, "trainer_log.jsonl")):
            logger.warning("Previous log file in this folder will be deleted.")
            os.remove(os.path.join(self.args.output_dir, "trainer_log.jsonl"))

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

        if hasattr(model, "pretrained_model"): # for models with valuehead (currently using LoRA only)
            backbone_model = getattr(model, "pretrained_model")
            torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))
        else:
            backbone_model = model

        if self.finetuning_args.finetuning_type == "lora":
            backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model))
        else: # freeze/full tuning
            backbone_model.config.use_cache = True
            backbone_model.save_pretrained(
                output_dir,
                state_dict=get_state_dict(backbone_model),
                safe_serialization=self.args.save_safetensors
            )
            backbone_model.config.use_cache = False
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

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
        backbone_model = getattr(model, "pretrained_model") if hasattr(model, "pretrained_model") else model

        if self.finetuning_args.finetuning_type == "lora":
            backbone_model.load_adapter(self.state.best_model_checkpoint, getattr(backbone_model, "active_adapter"))
            if hasattr(model, "v_head") and load_valuehead_params(model, self.state.best_model_checkpoint):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })
        else: # freeze/full-tuning
            load_trainable_params(backbone_model, self.state.best_model_checkpoint)
