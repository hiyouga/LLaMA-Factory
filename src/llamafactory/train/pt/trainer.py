# Copyright 2025 the LlamaFactory team.
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

from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments, ModelArguments

# ⚠️ add here
from collections import defaultdict
from typing import Dict

class CustomTrainer(Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)
        
        # ⚠️ add a _metrics_buffer to store lm_loss and Router_Loss
        self._metrics_buffer = defaultdict(lambda: defaultdict(lambda: torch.tensor(0.0)))

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        
        if outputs is not None:
            extra_metrics = {}
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    if k in ["lm_loss", "Router_loss"] and v is not None:
                        extra_metrics[k] = v
            else:
                if hasattr(outputs, "lm_loss"): extra_metrics["lm_loss"] = outputs.lm_loss
                if hasattr(outputs, "Router_loss"): extra_metrics["Router_loss"] = outputs.Router_loss

            if extra_metrics:
                mode = "train" if model.training else "eval"
                
                target_device = loss.device
                
                if self._metrics_buffer[mode]["steps"].device != target_device:
                     self._metrics_buffer[mode]["steps"] = self._metrics_buffer[mode]["steps"].to(target_device)

                with torch.no_grad():
                    for k, v in extra_metrics.items():
                        if self._metrics_buffer[mode][k].device != target_device:
                            self._metrics_buffer[mode][k] = self._metrics_buffer[mode][k].to(target_device)
                        
                        self._metrics_buffer[mode][k] += v.detach()
                    
                    self._metrics_buffer[mode]["steps"] += 1.0

        return (loss, outputs) if return_outputs else loss
    
    @override
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        仅当 Buffer 中有数据时，才计算并注入 Log。
        """
        if self._metrics_buffer["train"]["steps"] > 0:
            steps = self._metrics_buffer["train"]["steps"]
            for k, v in self._metrics_buffer["train"].items():
                if k == "steps": continue
                avg_val = v / steps
                logs[f"train/{k}"] = round(avg_val.item(), 4)
            
            # clear buffer
            self._metrics_buffer["train"].clear()
            self._metrics_buffer["train"]["steps"] = torch.tensor(0.0)

        super().log(logs, start_time=start_time)
    
    @override
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._metrics_buffer["eval"].clear()
        
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self._metrics_buffer["eval"]["steps"] > 0:
            steps = self._metrics_buffer["eval"]["steps"]
            for k, v in self._metrics_buffer["eval"].items():
                if k == "steps": continue
                
                # 分布式汇总 (All-Reduce Sum)
                if self.args.world_size > 1:
                    total_val = self.accelerator.reduce(v, reduction="sum")
                    total_steps = self.accelerator.reduce(steps, reduction="sum")
                    avg_val = total_val / total_steps
                else:
                    avg_val = v / steps
                
                metrics[f"{metric_key_prefix}_{k}"] = round(avg_val.item(), 4)
            
        return metrics
