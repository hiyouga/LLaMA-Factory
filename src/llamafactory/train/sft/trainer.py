# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import shutil
import torch.nn as nn
if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        all_channels: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        
        if os.path.exists(self.args.logging_dir):
            shutil.rmtree(self.args.logging_dir)
        self.writer = SummaryWriter(log_dir=self.args.logging_dir, flush_secs=120)   # TODO: 【√】draw channels loss

            
        self.all_channels = all_channels
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else torch.npu.device_count()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.finetuning_args.channel_loss:
            self.local_step_count = 0
            self.ch_steps_count = [0] * len(self.all_channels)
            self.ch_total_loss_list = [[] for _ in range(len(self.all_channels))]
            self.ch_token_count_list = [[] for _ in range(len(self.all_channels))]
            self._print_debug_info(f"[DEBUG] init channel loss related variables with {len(self.all_channels)} channels")

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
    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, trial: Union["optuna.Trial", dict[str, Any], None] = None, ignore_keys_for_eval: Optional[list[str]] = None, **kwargs):
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def compute_channel_loss(self, outputs, labels, inputs_channels, num_items_in_batch, position_ids=None):
        """
        精确统计每个channel的loss（token级别）
        模仿https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py中的channel_loss_func实现。
        改进了一下，loss其实就是累加，没必要把每个Token的都存，直接累加就行。
        """
        all_channels = self.all_channels
        assert all_channels is not None, 'No channels found, please check your dataset.'
        assert labels is not None, 'No labels found, please check the tokenizer?'
        
        logits = outputs.logits

        # 计算token级别loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        token_loss = loss_fct(flat_logits, flat_labels)
        mask = flat_labels != -100

        # 计算每个token属于哪个channel
        bs, seq = shift_labels.shape
        self.local_step_count += 1
        
        # 每个local step计算每个channel的总loss和token数量
        # 获取当前 batch 中实际出现的 channel
        import time
        start_time = time.time()
        channel_available = torch.zeros(bs, dtype=torch.bool, device=inputs_channels.device)
        for ch_index in torch.unique(inputs_channels):
            # 构造当前 channel 的可用性 mask
            # 对于每个样本，检查该 channel 是否可用
            for i in range(bs):
                channel_available[i] = ch_index in inputs_channels[i]
            
            # 扩展到所有 token：每个样本的 seq 个 token 都使用相同的可用性
            channel_token_mask = channel_available.repeat_interleave(seq)
            
            # 与 token 的有效性 mask 取与运算
            valid_channel_mask = channel_token_mask & mask
            
            if valid_channel_mask.any():
                valid_losses = token_loss[valid_channel_mask]
                total_loss = torch.sum(valid_losses).item()
                token_count = valid_losses.numel()
                if token_count > 0 and total_loss > 0.0:
                    self.ch_total_loss_list[ch_index].append(total_loss)
                    self.ch_token_count_list[ch_index].append(token_count)
                    self.ch_steps_count[ch_index] += 1
        end_time = time.time()
        self._print_debug_info(f"time cost: {end_time - start_time}")
        # 每个local step补0，方便计算。
        for ch_index in range(len(self.all_channels)):
            if len(self.ch_total_loss_list[ch_index]) < self.local_step_count:
                self.ch_total_loss_list[ch_index].append(0)
                self.ch_token_count_list[ch_index].append(0)
        # 每个梯度更新时候的出现过的channel的平均loss，可以计得和train step一样步数的channel loss
        if self.local_step_count % self.args.gradient_accumulation_steps == 0 and (self.state.global_step + 1) % self.args.logging_steps == 0:
            for ch_index, channel in enumerate(self.all_channels):
                # 取最末尾的gradient_accumulation_steps个值
                loss_sum = sum(self.ch_total_loss_list[ch_index][-self.args.gradient_accumulation_steps:])
                num_items = sum(self.ch_token_count_list[ch_index][-self.args.gradient_accumulation_steps:])
                ch_loss = loss_sum / (num_items + 1e-12)
                if ch_loss > 0.0:
                    self.writer.add_scalar(f"train/ch_loss_{channel}_global", ch_loss, self.state.global_step + 1)
        # 对每个channel，计算和train loss一样频率的channel loss(train loss是每gradient_accumulation_steps步计算一次)
        for ch_index, channel in enumerate(self.all_channels):
            if self.ch_steps_count[ch_index] % self.args.gradient_accumulation_steps == 0:
                loss_sum = 0
                num_items = 1e-12
                # 从ch_total_loss_list倒着找gradient_accumulation_steps个非0值
                for i in range(len(self.ch_total_loss_list[ch_index])-1, -1, -1):
                    if self.ch_total_loss_list[ch_index][i] > 0.0:
                        loss_sum += self.ch_total_loss_list[ch_index][i]
                        num_items += self.ch_token_count_list[ch_index][i]
                        if num_items >= self.args.gradient_accumulation_steps:
                            break
                ch_loss = loss_sum / (num_items + 1e-12)
                if ch_loss > 0.0:
                    self.writer.add_scalar(f"train/ch_loss_{channel}_local", ch_loss, self.ch_steps_count[ch_index] // self.args.gradient_accumulation_steps + 1)
        

        # 返回总loss，节约计算成本，不返回这个。
        # total_loss = token_loss.masked_select(mask).sum()
        # total_tokens = mask.sum()
        # return total_loss / num_items_in_batch if num_items_in_batch is not None \
        #     else total_loss / (total_tokens.float() + 1e-12)
    
    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        total_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        # 实现channel_loss
        if self.finetuning_args.channel_loss:
            inputs_channels = inputs.pop("channel", None)
            labels = inputs.pop("labels")
            self.compute_channel_loss(outputs, labels, inputs_channels, num_items_in_batch)
            

        return (total_loss, outputs) if return_outputs else total_loss

    def _print_debug_info(self, message):
        """多卡环境时, 在rank0打印
        """
        if dist.is_initialized():
            if self.is_local_process_zero():
                print(message)
        else:
            print(message)

    def _get_curr_gpu(self):
        """获取当前GPU
        """
        if dist.is_initialized():
            curr_gpu = dist.get_rank()
        else:
            curr_gpu = 0
        return curr_gpu

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
