# Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py

from typing import TYPE_CHECKING, List, Optional

from transformers import Seq2SeqTrainingArguments

from ...data import get_dataset, split_dataset
from ...extras.callbacks import FixValueHeadModelCallback
from ...extras.misc import fix_valuehead_checkpoint
from ...extras.ploting import plot_loss
from ...model import load_model_and_tokenizer
from ...train.rm.collator import PairwiseDataCollatorWithPadding
from ...train.rm.metric import compute_accuracy
from ...train.rm.trainer import PairwiseTrainer
from ...train.utils import create_modelcard_and_push


if TYPE_CHECKING:
    from transformers import TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_rm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    model, tokenizer = load_model_and_tokenizer(
        model_args, finetuning_args, training_args.do_train, add_valuehead=True
    )
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="rm")
    data_collator = PairwiseDataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Update arguments
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(remove_unused_columns=False))  # important for pairwise dataset
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks + [FixValueHeadModelCallback()],
        compute_metrics=compute_accuracy,
        **split_dataset(dataset, data_args, training_args),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
