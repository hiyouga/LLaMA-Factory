# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/ppo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorWithPadding

from ...data import get_dataset
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import fix_valuehead_checkpoint
from ..trainer_utils import create_ref_model, create_reward_model
from .trainer import CustomPPOTrainer


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
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset_module = get_dataset(model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

    tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
    reward_model = create_reward_model(model, model_args, finetuning_args)

    # Initialize our Trainer
    ppo_trainer: "CustomPPOTrainer" = CustomPPOTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks,
        model=model,
        reward_model=reward_model,
        ref_model=ref_model,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
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
