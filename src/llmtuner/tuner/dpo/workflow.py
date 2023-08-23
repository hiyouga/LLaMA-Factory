# Inspired by: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

from copy import deepcopy
from peft import PeftModel
from typing import TYPE_CHECKING, Optional, List
from transformers import Seq2SeqTrainingArguments

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.dpo.collator import DPODataCollatorWithPadding
from llmtuner.tuner.dpo.trainer import DPOPeftTrainer

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="rm")
    data_collator = DPODataCollatorWithPadding(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(remove_unused_columns=False)) # important for pairwise dataset
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = DPOPeftTrainer(
        finetuning_args=finetuning_args,
        ref_model=deepcopy(model) if not isinstance(model, PeftModel) else None,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args)
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
