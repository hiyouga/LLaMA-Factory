# Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py

from typing import TYPE_CHECKING, Optional, List
from transformers import Seq2SeqTrainingArguments

from llmtuner.data import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.callbacks import SavePeftModelCallback
from llmtuner.extras.ploting import plot_loss
from llmtuner.model import generate_model_card, load_model_and_tokenizer
from llmtuner.train.rm.collator import PairwiseDataCollatorWithPadding
from llmtuner.train.rm.metric import compute_accuracy
from llmtuner.train.rm.trainer import PairwiseTrainer

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments


def run_rm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="rm")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="rm")
    data_collator = PairwiseDataCollatorWithPadding(tokenizer, pad_to_multiple_of=4)

    # Update arguments
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(remove_unused_columns=False)) # important for pairwise dataset
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks + [SavePeftModelCallback()],
        compute_metrics=compute_accuracy,
        **split_dataset(dataset, data_args, training_args)
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
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
    if training_args.do_train:
        if training_args.push_to_hub:
            trainer.push_to_hub(**generate_model_card(model_args, data_args, finetuning_args))
        else:
            trainer.create_model_card(**generate_model_card(model_args, data_args, finetuning_args))
