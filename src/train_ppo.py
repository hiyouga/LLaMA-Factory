# coding=utf-8
# Implements parameter-efficient PPO training of fine-tuned models.
# This code is inspired by:
# https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py

import math

from torch.optim import AdamW
from transformers.optimization import get_scheduler
from trl import PPOConfig

from utils import (
    DynamicDataCollatorWithPadding,
    PPOPeftTrainer,
    LogCallback,
    load_pretrained,
    prepare_args,
    prepare_data,
    preprocess_data,
    plot_loss
)


def main():

    # Prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args(stage="ppo")
    dataset = prepare_data(model_args, data_args)
    model, tokenizer = load_pretrained(model_args, finetuning_args, training_args.do_train, stage="ppo")
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="ppo")
    data_collator = DynamicDataCollatorWithPadding(tokenizer)

    ppo_config = PPOConfig(
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        mini_batch_size=training_args.per_device_train_batch_size,
        batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=1,
        max_grad_norm=training_args.max_grad_norm
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=ppo_config.learning_rate)
    total_train_batch_size = \
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=(training_args.num_train_epochs * math.ceil(len(dataset) / total_train_batch_size))
    )

    # Initialize our Trainer
    ppo_trainer = PPOPeftTrainer(
        training_args=training_args,
        finetuning_args=finetuning_args,
        callbacks=[LogCallback()],
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    ppo_trainer.ppo_train(max_target_length=data_args.max_target_length)
    ppo_trainer.save_model()
    ppo_trainer.save_state() # must be after save_model
    if ppo_trainer.is_world_process_zero() and model_args.plot_loss:
        plot_loss(training_args.output_dir, keys=["loss", "reward"])


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
