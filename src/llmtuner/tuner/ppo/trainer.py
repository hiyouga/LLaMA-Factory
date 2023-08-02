import os
import math
import torch
from tqdm import tqdm
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from transformers import TrainerState, TrainerControl

from trl import PPOTrainer
from trl.core import LengthSampler

from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import AverageMeter, count_parameters, get_logits_processor
from llmtuner.tuner.core.trainer import PeftTrainer
from llmtuner.tuner.ppo.utils import cast_layernorm_dtype, replace_model

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from trl import AutoModelForCausalLMWithValueHead
    from llmtuner.extras.callbacks import LogCallback
    from llmtuner.hparams import FinetuningArguments


logger = get_logger(__name__)


class PPOPeftTrainer(PPOTrainer, PeftTrainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
        self,
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: List["LogCallback"],
        **kwargs
    ):
        PPOTrainer.__init__(self, **kwargs)
        self.args = training_args
        self.finetuning_args = finetuning_args
        self.log_callback = callbacks[0]
        self.state = TrainerState()
        self.control = TrainerControl()
        self._remove_log()

    def ppo_train(self, max_target_length: int) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        total_train_batch_size = (
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        len_dataloader = len(self.dataloader)
        num_examples = len(self.dataset)
        num_train_epochs = self.args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * len_dataloader)

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  Number of trainable parameters = {count_parameters(self.model)[0]}")

        # Keyword arguments for `model.generate`
        gen_kwargs = {
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "logits_processor": get_logits_processor()
        }
        length_sampler = LengthSampler(max_target_length // 2, max_target_length)
        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)

        dataiter = iter(self.dataloader)
        steps_trained = 0
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            batch = next(dataiter)
            steps_trained += 1

            # Cast to inference mode
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True

            # Get inputs
            queries, responses = self.get_inputs(batch, length_sampler, **gen_kwargs)
            rewards = self.get_rewards(queries, responses, unwrapped_model)

            # Cast to training mode
            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False

            # Run PPO step
            stats = self.step(queries, responses, rewards)
            loss_meter.update(stats["ppo/loss/total"], n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            self.state.global_step += 1
            self.log_callback.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step+1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / len_dataloader, 2)
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.log_callback.on_log(self.args, self.state, self.control)
                loss_meter.reset()
                reward_meter.reset()

            if (step+1) % self.args.save_steps == 0: # save checkpoint
                self.save_model(os.path.join(self.args.output_dir, f"checkpoint-{step+1}"))

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

            if steps_trained == len_dataloader:
                dataiter = iter(self.dataloader)
                steps_trained = 0

        self.log_callback.on_train_end(self.args, self.state, self.control)

    @torch.no_grad()
    def get_inputs(
        self,
        batch: Dict[str, torch.Tensor],
        length_sampler: Optional[Callable] = None,
        **generation_kwargs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""
        Generates model's responses given queries.
        """
        if length_sampler is not None:
            generation_kwargs["max_new_tokens"] = length_sampler()

        self.model, layer_norm_params = cast_layernorm_dtype(self.model)
        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        response: torch.Tensor = unwrapped_model.generate(**batch, **generation_kwargs)
        self.model, _ = cast_layernorm_dtype(self.model, layer_norm_params)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # Inspired by: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_seq2seq.py#L273
        if unwrapped_model.pretrained_model.generation_config._from_model_config:
            unwrapped_model.pretrained_model.generation_config._from_model_config = False

        queries, responses = [], []
        query, response = batch["input_ids"].detach().cpu(), response[:, batch["input_ids"].size(-1):].detach().cpu()
        for i in range(len(query)):
            query_length = (query[i] != self.tokenizer.pad_token_id).nonzero()[0]
            response_length = (response[i] != self.tokenizer.pad_token_id).nonzero()[-1] + 1
            queries.append(query[i, query_length:]) # remove padding from left
            responses.append(response[i, :response_length]) # remove padding from right

        return queries, responses

    @torch.no_grad()
    def get_rewards(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
        unwrapped_model: "AutoModelForCausalLMWithValueHead"
    ) -> List[torch.Tensor]:
        r"""
        Computes scores using given reward model.
        """
        replace_model(unwrapped_model, target="reward")
        batch = self.prepare_model_inputs(queries, responses)
        _, _, values = self.model(**batch, output_hidden_states=True, return_dict=True)
        rewards = [reward for reward in values[:, -1].float().detach().cpu()] # use fp32 type
        replace_model(unwrapped_model, target="default")
        return rewards

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            self._save(output_dir)
