import os
import math
import torch
from tqdm import tqdm
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from transformers import TrainerState, TrainerControl

from trl import PPOTrainer
from trl.core import LengthSampler, PPODecorators, logprobs_from_logits

from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import AverageMeter, count_parameters, get_logits_processor
from llmtuner.tuner.core.trainer import PeftTrainer
from llmtuner.tuner.ppo.utils import cast_layernorm_dtype, replace_model

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from trl import AutoModelForCausalLMWithValueHead
    from llmtuner.extras.callbacks import LogCallback
    from llmtuner.hparams import FinetuningArguments, GeneratingArguments


logger = get_logger(__name__)


class PPOPeftTrainer(PPOTrainer, PeftTrainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
        self,
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: List["LogCallback"],
        compute_dtype: torch.dtype,
        **kwargs
    ):
        PPOTrainer.__init__(self, **kwargs)
        self.args = training_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        self.log_callback = callbacks[0]
        self.compute_dtype = compute_dtype
        self.state = TrainerState()
        self.control = TrainerControl()

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
        gen_kwargs = self.generating_args.to_dict()
        gen_kwargs["eos_token_id"] = list(set([self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids))
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()

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

        self.model, layer_norm_params = cast_layernorm_dtype(self.model, self.compute_dtype)
        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        response: torch.Tensor = unwrapped_model.generate(**batch, **generation_kwargs)
        self.model, _ = cast_layernorm_dtype(self.model, self.compute_dtype, layer_norm_params)

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

        with torch.cuda.amp.autocast(dtype=self.compute_dtype): # support bf16
            _, _, values = self.model(**batch, output_hidden_states=True, return_dict=True)

        if values.size(0) != batch["input_ids"].size(0): # adapt to chatglm2
            values = torch.transpose(values, 0, 1)

        rewards = [reward for reward in values[:, -1].float().detach().cpu()] # use fp32 type
        replace_model(unwrapped_model, target="default")
        return rewards

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: Optional[bool] = False,
        response_masks: Optional[torch.Tensor] = None
    ):
        r"""
        Calculates model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
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

            with torch.cuda.amp.autocast(dtype=self.compute_dtype): # support bf16
                logits, _, values = model(**input_kwargs)

            if values.size(0) != input_ids.size(0): # adapt to chatglm2
                values = torch.transpose(values, 0, 1)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat(
                        (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                    )[1:]

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

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            self._save(output_dir)
