import os
import math
import torch
from tqdm import tqdm
from typing import Callable, Dict, List, Literal, Optional, Tuple

from transformers import Seq2SeqTrainingArguments, TrainerState
from transformers.modeling_utils import PreTrainedModel

from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from .peft_trainer import PeftTrainer, LogCallback

from .config import FinetuningArguments

from .other import (
    AverageMeter,
    get_logger,
    get_logits_processor
)


logger = get_logger(__name__)


def replace_model(model: AutoModelForCausalLMWithValueHead, target: Literal["default", "reward"]) -> None:
    if target == "reward": # save original head temporarily
        valuehead_state_dict = model.v_head.state_dict()

        setattr(model, "origin_head_weight", valuehead_state_dict["summary.weight"])
        setattr(model, "origin_head_bias", valuehead_state_dict["summary.bias"])

    model.pretrained_model.set_adapter(target) # set the LoRA adapter to be active
    model.v_head.load_state_dict({
        "summary.weight": getattr(model, "{}_head_weight".format(target)),
        "summary.bias": getattr(model, "{}_head_bias".format(target))
    })


def cast_layernorm_dtype(
        model: AutoModelForCausalLMWithValueHead,
        layer_norm_names: List[str] = ["norm", "ln_f"], # for LLaMA and BLOOM setting
        layer_norm_params: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[AutoModelForCausalLMWithValueHead, Dict[str, torch.Tensor]]:

    layer_norm_state_dict = {}

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            if layer_norm_params is not None:
                param.data = layer_norm_params[name] # restore float32 weights
            else:
                layer_norm_state_dict[name] = param.data.detach().clone() # store float32 weights for stability
                param.data = param.data.to(torch.float16)

    return model, layer_norm_state_dict


class PPOPeftTrainer(PPOTrainer, PeftTrainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
            self,
            training_args: Seq2SeqTrainingArguments,
            finetuning_args: FinetuningArguments,
            callbacks: List[LogCallback],
            **kwargs
    ):
        PPOTrainer.__init__(self, **kwargs)
        self.args = training_args
        self.finetuning_args = finetuning_args
        self.log_callback = callbacks[0]
        self.state = TrainerState()
        self.data_collator = self.accelerator.prepare(kwargs["data_collator"]) # override the data collator of PPOTrainer

    def ppo_train(self, max_target_length: int) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        total_train_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps * self.args.world_size
        len_dataloader = len(self.dataloader)
        num_steps_per_epoch = max(len_dataloader // self.config.gradient_accumulation_steps, 1)
        num_examples = len(self.dataset)
        num_train_epochs = self.args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * num_steps_per_epoch)

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.config.batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  Number of trainable parameters = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # Keyword arguments for `model.generate`
        gen_kwargs = {
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "logits_processor": get_logits_processor()
        }
        output_length_sampler = LengthSampler(max_target_length // 2, max_target_length)
        unwrapped_model: PreTrainedModel = self.accelerator.unwrap_model(self.model)

        dataiter = iter(self.dataloader)
        steps_trained = 0
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()

        for step in tqdm(range(max_steps), disable=not self.is_world_process_zero()):

            for _ in range(self.config.gradient_accumulation_steps):

                batch = next(dataiter)
                steps_trained += 1

                unwrapped_model.gradient_checkpointing_disable()
                unwrapped_model.config.use_cache = True

                # Get response from model
                query_tensors: torch.Tensor = batch["input_ids"]
                response_tensors = self.generate(batch, length_sampler=output_length_sampler, return_prompt=False, **gen_kwargs)

                queries: List[torch.Tensor] = []
                responses: List[torch.Tensor] = []
                for i in range(len(query_tensors)):
                    query_length = (query_tensors[i] != self.tokenizer.pad_token_id).nonzero()[0]
                    response_length = (response_tensors[i] != self.tokenizer.pad_token_id).nonzero()[-1] + 1
                    queries.append(query_tensors[i, query_length:]) # remove padding from left
                    if response_length < 2: # make response have at least 2 tokens
                        responses.append(response_tensors.new_empty(2).fill_(self.tokenizer.eos_token_id))
                    else:
                        responses.append(response_tensors[i, :response_length]) # remove padding from right

                # Compute rewards
                replace_model(unwrapped_model, target="reward")
                _, _, values = self.model(**self.prepare_model_inputs(queries, responses))
                rewards = [reward for reward in values[:, -1].to(torch.float32)] # use float32 type
                replace_model(unwrapped_model, target="default") # make sure the model is default at the end

                # Run PPO step
                unwrapped_model.gradient_checkpointing_enable()
                unwrapped_model.config.use_cache = False

                stats = self.step(queries, responses, rewards)

                loss_meter.update(stats["ppo/loss/total"], n=len(rewards))
                reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

                if steps_trained == len_dataloader:
                    dataiter = iter(self.dataloader)
                    steps_trained = 0

            if self.is_world_process_zero() and (step+1) % self.args.logging_steps == 0:
                logs = {
                    "loss": round(loss_meter.avg, 4),
                    "reward": round(reward_meter.avg, 4),
                    "learning_rate": stats["ppo/learning_rate"],
                    "epoch": round(step / num_steps_per_epoch, 2)
                }
                print(logs)
                logs["step"] = step
                self.state.log_history.append(logs)
                self.log_callback.on_log(self.args, self.state, None)
                loss_meter.reset()
                reward_meter.reset()

            if (step+1) % self.args.save_steps == 0: # save checkpoint
                self.save_model(os.path.join(self.args.output_dir, f"checkpoint-{step+1}"))

    @torch.no_grad()
    def generate(
            self,
            inputs: Dict[str, torch.Tensor],
            length_sampler: Optional[Callable] = None,
            return_prompt: Optional[bool] = True,
            **generation_kwargs,
    ) -> torch.Tensor:
        r"""
        Generates model's responses given queries.

        Subclass and override to inject custom behavior.
        """
        self.model, layer_norm_params = cast_layernorm_dtype(self.model)

        if length_sampler is not None:
            generation_kwargs["max_new_tokens"] = length_sampler()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        response = unwrapped_model.generate(**inputs, **generation_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # Inspired by: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_seq2seq.py#L273
        if unwrapped_model.pretrained_model.generation_config._from_model_config:
            unwrapped_model.pretrained_model.generation_config._from_model_config = False

        self.model, _ = cast_layernorm_dtype(self.model, layer_norm_params)

        if not return_prompt and not self.is_encoder_decoder:
            return response[:, inputs["input_ids"].size(1):]
        return response

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            self._save(output_dir)
