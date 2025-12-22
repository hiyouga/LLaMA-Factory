# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
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

import math
import os
import sys
import warnings
from importlib.metadata import version as get_pkg_version
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, cast

from packaging.version import Version
import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor, torch_gc
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm, logprobs_from_logits, masked_mean, masked_whiten


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class LlamaFactoryValueModelAdapter(torch.nn.Module):
    r"""
    A specific value model adapter to conform to the TRL interface.
    """

    def __init__(self, model: "AutoModelForCausalLMWithValueHead"):
        super().__init__()
        self.wrapped_model = model
        self.v_head = model.v_head
        self.base_model_prefix = model.base_model_prefix

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def score(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        return self.v_head(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_model, name)


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""Inherit PPOTrainer."""

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

        try:
            trl_version = Version(get_pkg_version("trl"))
        except Exception:
            trl_version = Version("0")

        if trl_version >= Version("0.24.0"):
            local_dataloader_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size
            num_mini_batches = training_args.gradient_accumulation_steps * finetuning_args.ppo_buffer_size
            ppo_config = PPOConfig(
                learning_rate=training_args.learning_rate,
                per_device_train_batch_size=local_dataloader_batch_size,
                gradient_accumulation_steps=1,
                num_mini_batches=num_mini_batches,
                num_ppo_epochs=finetuning_args.ppo_epochs,
                max_grad_norm=training_args.max_grad_norm,
                seed=training_args.seed,
                whiten_rewards=finetuning_args.ppo_whiten_rewards,
                fp16=training_args.fp16,  # TODO to be fix
                bf16=training_args.bf16,
            )
        else:
            ppo_config = PPOConfig(
                learning_rate=training_args.learning_rate,
                mini_batch_size=training_args.per_device_train_batch_size,
                batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                num_ppo_epochs=finetuning_args.ppo_epochs,
                max_grad_norm=training_args.max_grad_norm,
                seed=training_args.seed,
                whiten_rewards=finetuning_args.ppo_whiten_rewards,
                fp16=training_args.fp16,  # TODO to be fix
                bf16=training_args.bf16,
            )

        # Keep backward-compatible attributes for the custom PPO loop.
        ppo_config.batch_size = backward_batch_size * finetuning_args.ppo_buffer_size
        ppo_config.mini_batch_size = training_args.per_device_train_batch_size

        # Normalize logging config across TRL versions.
        report_to = getattr(training_args, "report_to", None)
        if hasattr(ppo_config, "report_to"):
            ppo_config.report_to = report_to
        ppo_config.log_with = None if report_to in (None, "none", ["none"], []) else report_to
        ppo_config.logging_steps = training_args.logging_steps
        ppo_config.save_steps = training_args.save_steps

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None and hasattr(ppo_config, "accelerator_kwargs"):
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )
        ppo_config.total_episodes = num_training_steps * backward_batch_size

        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
        )
        if not isinstance(lr_scheduler, LambdaLR):
            raise ValueError(
                "The PPO trainer requires lr_scheduler to be torch.optim.lr_scheduler.LambdaLR, "
                f"got {type(lr_scheduler)}."
            )
        lr_scheduler = cast(LambdaLR, lr_scheduler)

        self.generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        # Force policy model to have generation_config if missing (for TRL logic)
        if not hasattr(model, "generation_config"):
            model.generation_config = self.generation_config

        # Force policy model to have is_gradient_checkpointing if missing (for TRL 0.24)
        if not hasattr(model, "is_gradient_checkpointing"):
            if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "is_gradient_checkpointing"):
                model.is_gradient_checkpointing = model.pretrained_model.is_gradient_checkpointing
            else:
                model.is_gradient_checkpointing = getattr(model, "gradient_checkpointing", False)

        reward_model_for_trl = reward_model
        if reward_model_for_trl is None or isinstance(reward_model_for_trl, str):
            reward_model_for_trl = torch.nn.Identity()

        PPOTrainer.__init__(
            self,
            args=ppo_config,
            processing_class=tokenizer,
            model=model,
            ref_model=ref_model,
            reward_model=reward_model_for_trl,
            train_dataset=train_dataset,
            value_model=LlamaFactoryValueModelAdapter(model),
            data_collator=data_collator,
            optimizers=(optimizer, lr_scheduler),
        )

        self.train_args = training_args
        self.config = ppo_config  # Save PPOConfig as self.config for compatibility
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        if self.reward_model is not None and not isinstance(self.reward_model, str):
            if getattr(self.reward_model.config, "pad_token_id", None) is None:
                self.reward_model.config.pad_token_id = tokenizer.pad_token_id
            if not hasattr(self.reward_model, "generation_config"):
                self.reward_model.generation_config = self.generation_config
        self.current_device = get_current_device()  # patch for deepspeed training

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.train_args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""Implement training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer."""
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.train_args.per_device_train_batch_size
            * self.train_args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.train_args.world_size
        )
        if self.train_args.max_steps > 0:
            num_examples = total_train_batch_size * self.train_args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.train_args.max_steps
            steps_in_epoch = self.train_args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.train_args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.train_args.per_device_train_batch_size:,}")
        logger.info_rank0(
            f"  Total train batch size (w. parallel, buffer, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.train_args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.train_args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Get inputs
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch = {
                    "input_ids": batch["input_ids"][idx : idx + self.config.mini_batch_size],
                    "attention_mask": batch["attention_mask"][idx : idx + self.config.mini_batch_size],
                }
                mini_batch_queries, mini_batch_responses = self.get_inputs(mini_batch)
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

            # Run PPO step
            self.model.train()
            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.train_args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.train_args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.train_args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.train_args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.train_args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.train_args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.train_args, self.state, self.control)

    def prepare_model_inputs(self, queries: list["torch.Tensor"], responses: list["torch.Tensor"]) -> dict[str, "torch.Tensor"]:
        queries = [q.to(self.current_device) for q in queries]
        responses = [r.to(self.current_device) for r in responses]

        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def step(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
        rewards: list["torch.Tensor"],
    ) -> dict[str, Any]:

        # 1. Prepare batch
        model_inputs = self.prepare_model_inputs(queries, responses)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        rewards = torch.tensor(rewards, device=self.current_device)

        bs = len(queries)

        # 2. Compute logprobs, values, ref_logprobs
        with torch.no_grad():
            logprobs_list, values_list, ref_logprobs_list = [], [], []

            fbs = self.config.mini_batch_size # forward batch size
            for i in range(math.ceil(bs / fbs)):
                start, end = i * fbs, (i + 1) * fbs
                mb_input_ids = input_ids[start:end]
                mb_attention_mask = attention_mask[start:end]

                # Forward pass
                output, vpred_temp = self.model(input_ids=mb_input_ids, attention_mask=mb_attention_mask, return_dict=True, use_cache=False)
                logits = output.logits[:, :-1]
                logits /= self.config.temperature + 1e-7
                all_logprobs = logprobs_from_logits(logits, mb_input_ids[:, 1:])
                vpred = vpred_temp[:, :-1].squeeze(-1)

                # Reference model
                if self.ref_model is None:
                    with self.null_ref_context():
                        policy_model = getattr(self.model, "policy", self.model)
                        ref_output = policy_model(
                            input_ids=mb_input_ids,
                            attention_mask=mb_attention_mask,
                            return_dict=True,
                            use_cache=False,
                        )
                else:
                    ref_output = self.ref_model(input_ids=mb_input_ids, attention_mask=mb_attention_mask, return_dict=True, use_cache=False)

                ref_logits = ref_output.logits[:, :-1]
                ref_logits /= self.config.temperature + 1e-7
                ref_logprobs = logprobs_from_logits(ref_logits, mb_input_ids[:, 1:])

                logprobs_list.append(all_logprobs)
                values_list.append(vpred)
                ref_logprobs_list.append(ref_logprobs)

            logprobs = torch.cat(logprobs_list)
            values = torch.cat(values_list)
            ref_logprobs = torch.cat(ref_logprobs_list)

            # Construct masks
            masks = torch.zeros_like(logprobs)
            for j in range(len(queries)):
                q_len = len(queries[j])
                r_len = len(responses[j])
                masks[j, q_len-1 : q_len + r_len - 1] = 1.0

            padding_mask = (input_ids[:, 1:] == self.tokenizer.pad_token_id)
            masks = masks * (~padding_mask)

        # 3. Compute rewards, advantages
        logr = ref_logprobs - logprobs

        # KL penalty
        if self.config.kl_estimator == "k1":
            kl = -logr
        else:
            kl = (logr.exp() - 1) - logr

        non_score_reward = -self.config.kl_coef * kl

        # Add scores to rewards
        full_rewards = non_score_reward.clone()
        for j in range(len(queries)):
            q_len = len(queries[j])
            r_len = len(responses[j])
            idx = q_len + r_len - 2
            if idx < 0: idx = 0
            full_rewards[j, idx] += rewards[j]

        # Whiten
        if self.config.whiten_rewards:
            full_rewards = masked_whiten(full_rewards, mask=masks, shift_mean=False)
            full_rewards = full_rewards * masks

        # Advantages (GAE)
        lastgaelam = 0
        advantages_reversed = []
        gen_len = values.shape[1]
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = full_rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], axis=1)
        returns = advantages + values

        # Whiten advantages
        advantages = masked_whiten(advantages, masks)
        advantages = advantages * masks

        # 4. PPO Epochs
        approxkl_stats = []
        pg_loss_stats = []
        vf_loss_stats = []
        entropy_stats = []

        for _ in range(self.config.num_ppo_epochs):
            b_inds = torch.randperm(bs)
            for mini_batch_start in range(0, bs, self.config.mini_batch_size):
                mini_batch_end = mini_batch_start + self.config.mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]

                mb_input_ids = input_ids[mini_batch_inds]
                mb_attention_mask = attention_mask[mini_batch_inds]
                mb_advantages = advantages[mini_batch_inds]
                mb_returns = returns[mini_batch_inds]
                mb_values = values[mini_batch_inds]
                mb_logprobs = logprobs[mini_batch_inds]
                mb_masks = masks[mini_batch_inds]

                output, vpred_temp = self.model(input_ids=mb_input_ids, attention_mask=mb_attention_mask, return_dict=True, use_cache=False)
                logits = output.logits[:, :-1]
                logits /= self.config.temperature + 1e-7
                new_logprobs = logprobs_from_logits(logits, mb_input_ids[:, 1:])
                vpred = vpred_temp[:, :-1].squeeze(-1)

                vpredclipped = torch.clamp(
                    vpred,
                    mb_values - self.config.cliprange_value,
                    mb_values + self.config.cliprange_value,
                )

                vf_losses1 = torch.square(vpred - mb_returns)
                vf_losses2 = torch.square(vpredclipped - mb_returns)
                vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mb_masks)

                logprobs_diff = new_logprobs - mb_logprobs
                ratio = torch.exp(logprobs_diff)
                pg_losses = -mb_advantages * ratio
                pg_losses2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
                pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mb_masks)

                loss = pg_loss + self.config.vf_coef * vf_loss

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                     approxkl = 0.5 * masked_mean((logprobs_diff**2), mb_masks)
                     probabilities = torch.softmax(logits, dim=-1)
                     entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probabilities * logits, dim=-1)
                     approxkl_stats.append(approxkl)
                     pg_loss_stats.append(pg_loss)
                     vf_loss_stats.append(vf_loss)
                     entropy_stats.append(masked_mean(entropy, mb_masks))

        return {
            "ppo/loss/total": torch.stack(pg_loss_stats).mean() + self.config.vf_coef * torch.stack(vf_loss_stats).mean(),
            "ppo/policy/approxkl": torch.stack(approxkl_stats).mean().item(),
            "ppo/learning_rate": self.lr_scheduler.get_last_lr()[0],
        }

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: dict[str, "torch.Tensor"]) -> tuple[list["torch.Tensor"], list["torch.Tensor"]]:
        r"""Generate model's responses given queries."""
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(self.model, self.accelerator):
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            policy_model = getattr(unwrapped_model, "policy", unwrapped_model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(policy_model)

            generate_output: torch.Tensor = policy_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(policy_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right

        return queries, responses

    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        if self.finetuning_args.reward_model_type == "api":
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
            return get_rewards_from_server(self.reward_model, messages)

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)

        if hasattr(unwrapped_model, "value_model"):
             value_model_obj = unwrapped_model.value_model
        else:
             value_model_obj = unwrapped_model

        if self.finetuning_args.reward_model_type in ["lora", "oft"]:
            replace_model(value_model_obj, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
            values: torch.Tensor = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type in ["lora", "oft"]:
            replace_model(value_model_obj, target="default")

        rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()  # use fp32 type

    @override
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        torch_gc()

        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        r"""Save model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.train_args.output_dir

        # TRL 0.24 compatibility: save policy only
        backup_model = None
        backup_deepspeed = None
        if hasattr(self.model, "policy"): # Check if wrapper
             backup_model = self.model
             self.model = self.model.policy

             if self.is_deepspeed_enabled:
                backup_deepspeed = getattr(self, "deepspeed", None)
                self.deepspeed = self.model

        try:
            if self.is_fsdp_enabled or self.is_deepspeed_enabled:
                try:
                    state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                    if self.train_args.should_save:
                        self._save(output_dir, state_dict=state_dict)
                except ValueError:
                    logger.warning_rank0(
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                        " use zero_to_fp32.py to recover weights"
                    )
                    if self.train_args.should_save:
                        self._save(output_dir, state_dict={})
                    # remove the dummy state_dict
                    remove_dummy_checkpoint(self.train_args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                    self.model.save_checkpoint(output_dir)

            elif self.train_args.should_save:
                unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
                self._save(output_dir, state_dict=unwrapped_model.state_dict())
        finally:
             if backup_model is not None:
                 self.model = backup_model
                 if self.is_deepspeed_enabled:
                      self.deepspeed = backup_deepspeed
