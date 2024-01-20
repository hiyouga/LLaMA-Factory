# Inspired by: https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py

import math
from typing import TYPE_CHECKING, List, Optional

from torch.optim import AdamW
from transformers import DataCollatorWithPadding
from transformers.optimization import get_scheduler
from trl import PPOConfig

from ...data import get_dataset
from ...extras.callbacks import FixValueHeadModelCallback
from ...extras.misc import fix_valuehead_checkpoint
from ...extras.ploting import plot_loss
from ...model import load_model_and_tokenizer
from ...train.ppo.trainer import CustomPPOTrainer
from ...train.utils import create_ref_model, create_reward_model


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_ppo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    model, tokenizer = load_model_and_tokenizer(
        model_args, finetuning_args, training_args.do_train, add_valuehead=True
    )
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="ppo")

    tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
    reward_model = create_reward_model(model, model_args, finetuning_args)

    # Create ppo config
    backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    ppo_config = PPOConfig(
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        mini_batch_size=training_args.per_device_train_batch_size,
        batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=finetuning_args.ppo_epochs,
        max_grad_norm=training_args.max_grad_norm,
        seed=training_args.seed,
        optimize_device_cache=True,
        target=finetuning_args.ppo_target,
        log_with=finetuning_args.ppo_logger,
        use_score_scaling=finetuning_args.ppo_score_norm,
        use_score_norm=finetuning_args.ppo_score_norm,
        whiten_rewards=finetuning_args.ppo_whiten_rewards,
        accelerator_kwargs={"step_scheduler_with_optimizer": False},
    )

    # Create optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=training_args.learning_rate)
    if training_args.max_steps > 0:
        num_training_steps = training_args.max_steps
    else:
        total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
        num_training_steps = training_args.num_train_epochs * math.ceil(len(dataset) / total_train_batch_size)

    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
        num_training_steps=num_training_steps,
    )

    # Initialize our Trainer
    ppo_trainer = CustomPPOTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks + [FixValueHeadModelCallback()],
        reward_model=reward_model,
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Training
    if training_args.do_train:
        ppo_trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        ppo_trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)
        ppo_trainer.save_state()  # must be called after save_model to have a folder
        if ppo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
