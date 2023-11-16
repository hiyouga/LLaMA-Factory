# Inspired by: https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py

import math
from trl import PPOConfig
from torch.optim import AdamW
from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorWithPadding
from transformers.optimization import get_scheduler

from llmtuner.data import get_dataset, preprocess_dataset
from llmtuner.extras.callbacks import SavePeftModelCallback
from llmtuner.extras.ploting import plot_loss
from llmtuner.model import load_model_and_tokenizer
from llmtuner.train.utils import create_ref_model, create_reward_model
from llmtuner.train.ppo.trainer import CustomPPOTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


def run_ppo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="ppo")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="ppo")

    tokenizer.padding_side = "left" # use left-padding in generation while using right-padding in training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, stage="ppo")
    reward_model = create_reward_model(model, model_args, finetuning_args)

    # Create ppo config
    ppo_config = PPOConfig(
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        mini_batch_size=training_args.per_device_train_batch_size,
        batch_size=training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=1,
        max_grad_norm=training_args.max_grad_norm,
        seed=training_args.seed,
        optimize_device_cache=True,
        target=finetuning_args.ppo_target,
        log_with=finetuning_args.ppo_logger,
        use_score_scaling=finetuning_args.ppo_score_norm,
        use_score_norm=finetuning_args.ppo_score_norm,
        whiten_rewards=finetuning_args.ppo_whiten_rewards,
        accelerator_kwargs={"step_scheduler_with_optimizer": False}
    )

    # Create optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=training_args.learning_rate)
    if training_args.max_steps > 0:
        num_training_steps = training_args.max_steps
    else:
        total_train_batch_size = (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        )
        num_training_steps = training_args.num_train_epochs * math.ceil(len(dataset) / total_train_batch_size)

    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
        num_training_steps=num_training_steps
    )

    # Initialize our Trainer
    ppo_trainer = CustomPPOTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks + [SavePeftModelCallback()],
        reward_model=reward_model,
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    # Training
    if training_args.do_train:
        ppo_trainer.ppo_train()
        ppo_trainer.save_model()
        ppo_trainer.save_state() # must be called after save_model to have a folder
        if ppo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
