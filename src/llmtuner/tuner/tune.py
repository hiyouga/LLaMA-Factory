from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llmtuner.extras.callbacks import LogCallback
from llmtuner.tuner.core import get_train_args, load_model_and_tokenizer
from llmtuner.tuner.pt import run_pt
from llmtuner.tuner.sft import run_sft
from llmtuner.tuner.rm import run_rm
from llmtuner.tuner.ppo import run_ppo

if TYPE_CHECKING:
    from transformers import TrainerCallback


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, general_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    if general_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif general_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, callbacks)
    elif general_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif general_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, callbacks)


def export_model(args: Optional[Dict[str, Any]] = None, max_shard_size: Optional[str] = "10GB"):
    model_args, _, training_args, finetuning_args, _ = get_train_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    model.save_pretrained(training_args.output_dir, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    run_exp()
