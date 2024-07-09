# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling

from ...data import get_dataset, split_dataset, SeqParallelDataCollatorForLanguageModeling
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import CustomTrainer, CustomSeqParallelTrainer

import os
import torch
from ...easy_context import apply_seq_parallel_monkey_patch

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="pt", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    apply_seq_parallel_monkey_patch(finetuning_args.parallel_mode, "llama")

    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    local_rank = int(os.getenv("LOCAL_RANK"))
    print(f"seq_len: {data_args.cutoff_len}")
    
    data_collator = SeqParallelDataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        seq_algo=finetuning_args.parallel_mode,
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
        device=torch.device("cuda", local_rank)
    )

    # Initialize our Trainer
    # trainer = CustomTrainer(
    #     model=model,
    #     args=training_args,
    #     finetuning_args=finetuning_args,
    #     data_collator=data_collator,
    #     callbacks=callbacks,
    #     **tokenizer_module,
    #     **split_dataset(dataset, data_args, training_args),
    # )

    trainer = CustomSeqParallelTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **tokenizer_module,
        **split_dataset(dataset, data_args, training_args),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
    
    # Evaluation
    # if training_args.do_eval:
    #     metrics = trainer.evaluate(metric_key_prefix="eval")
    #     try:
    #         perplexity = math.exp(metrics["eval_loss"])
    #     except OverflowError:
    #         perplexity = float("inf")

    #     metrics["perplexity"] = perplexity
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
