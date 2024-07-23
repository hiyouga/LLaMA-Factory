import json
import os
import sys
import math
import warnings
from types import MethodType
import torch
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

# Define logger
import logging
logger = logging.getLogger(__name__)

class CustomPPOv2Trainer(PPOv2Trainer, Trainer):
    def __init__(
        self,
        model_args,
        training_args,
        finetuning_args,
        generating_args,
        callbacks,
        model,
        reward_model,
        ref_model,
        tokenizer,
        processor,
        dataset,
        data_collator,
    ):
        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOv2Config(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            target=finetuning_args.ppo_target,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with == "tensorboard":
                ppo_config.log_with = None

        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(len(dataset) / total_train_batch_size)

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def ppo_train(self, resume_from_checkpoint=None):
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info("  Num examples = {:,}".format(num_examples))
            logger.info("  Num Epochs = {:,}".format(num_train_epochs))
            logger.info("  Instantaneous batch size per device = {:,}".format(self.args.per_device_train_batch_size))
            logger.info(
                "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {:,}".format(
                    total_train_batch_size
                )
            )
            logger.info("  Gradient Accumulation steps = {:,}".format(self.args.gradient_accumulation_steps))
            logger.info("  Num optimization epochs per batch = {:,}".format(self.finetuning_args.ppo_epochs))
            logger.info("  Total training steps = {:,}".format(max_steps))
            logger.info("  Number of trainable parameters = {:,}".format(count_parameters(self.model)[0]))

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            self.model.eval()
            self.tokenizer.padding_side = "right"
            queries, responses, rewards = [], [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch_queries, mini_batch_responses = self.get_inputs(
                    batch[idx : idx + self.config.mini_batch_size]
                )
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

            self.model.train()
            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"]
                    batch["response"] = responses
                    batch["reward"] = rewards
                    self.accelerator.log(batch, step=step)
                except Exception as e:
                    logger.warning(f"Logging failed at step {step}: {e}")

            self.state.global_step += 1
            self.state.log_history.append(stats)

            self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss=loss_meter.avg, model=self.model, trial=None, epoch=None)

            if self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)
        self.state.is_world_process_zero = self.is_world_process_zero()
        if self.is_world_process_zero():
            self._save_training_summary()

        return loss_meter.avg

    def get_inputs(self, batch):
        """
        Prepares inputs for the model.
        """
        inputs = self.tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.current_device) for k, v in inputs.items()}
        return inputs

    def get_rewards(self, queries, responses):
        """
        Calculates rewards for the generated responses.
        """
        with torch.no_grad():
            rewards = []
            for query, response in zip(queries, responses):
                reward = self.reward_model(query, response)
                rewards.append(reward)
        return rewards

    def create_optimizer(self, model, training_args, finetuning_args):
        """
        Creates optimizer for training.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
        return optimizer

    def create_scheduler(self, training_args, num_training_steps, optimizer):
        """
        Creates scheduler for learning rate adjustment.
        """
        scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        """
        Performs logging, saving and evaluation if needed.
        """
        if self.control.should_log:
            logs = {}
            logs["loss"] = tr_loss
            self.state.log_history.append(logs)
            self.control = self.callback_handler.on_log(self.args, self.state, self.control)
        
        if self.control.should_save:
            self._save_checkpoint(self.model, trial, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control)

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Saves model checkpoint.
        """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_folder)

        self.store_flos()
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save(state_dict, os.path.join(checkpoint_dir, WEIGHTS_NAME))
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as f:
            json.dump(self.state_dict(), f)

        if metrics is not None and self.is_world_process_zero():
            with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)

        self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def evaluate(self):
        """
        Evaluation logic.
        """
        # Custom evaluation logic
        return {"eval_loss": 0.0}  # Replace with actual evaluation metrics

def get_current_device():
    """
    Get the current device (CPU or GPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    """
    Count the number of parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params

class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FixValueHeadModelCallback:
    """
    A callback to fix value head model during PPOv2 training.
    """
    def on_step_end(self, args, state, control, **kwargs):
        # Custom logic for fixing value head model
        return control

class SaveProcessorCallback:
    """
    A callback to save processor during PPOv2 training.
    """
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        # Custom logic for saving processor
        return control
