import os
import json
import time
from datetime import timedelta

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments
)
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class LogCallback(TrainerCallback):

    def __init__(self, runner=None):
        self.runner = runner
        self.start_time = time.time()
        self.tracker = {}

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        r"""
        Event called at the beginning of training.
        """
        self.start_time = time.time()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        r"""
        Event called at the beginning of a training step. If using gradient accumulation, one training step
        might take several inputs.
        """
        if self.runner is not None and self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        r"""
        Event called at the end of an substep during gradient accumulation.
        """
        if self.runner is not None and self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        r"""
        Event called after logging the last logs.
        """
        if not state.is_world_process_zero:
            return

        cur_time = time.time()
        cur_steps = state.log_history[-1].get("step")
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_steps = state.max_steps - cur_steps
        remaining_time = remaining_steps * avg_time_per_step
        self.tracker = {
            "current_steps": cur_steps,
            "total_steps": state.max_steps,
            "loss": state.log_history[-1].get("loss", None),
            "eval_loss": state.log_history[-1].get("eval_loss", None),
            "predict_loss": state.log_history[-1].get("predict_loss", None),
            "reward": state.log_history[-1].get("reward", None),
            "learning_rate": state.log_history[-1].get("learning_rate", None),
            "epoch": state.log_history[-1].get("epoch", None),
            "percentage": round(cur_steps / state.max_steps * 100, 2) if state.max_steps != 0 else 100,
            "elapsed_time": str(timedelta(seconds=int(elapsed_time))),
            "remaining_time": str(timedelta(seconds=int(remaining_time)))
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "trainer_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(self.tracker) + "\n")
