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
        channel_index_map: Optional[dict[str, Any]] = None,
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

            
        self.channel_index_map = channel_index_map
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
            self.all_channels = [v for k, v in self.channel_index_map.items()]
            self.cumulative_dict = {
                "local_step_count": 0,
                **{f"{v}_ch_steps_count": 0 for k, v in self.channel_index_map.items()},
                **{f"{v}_total_loss_list": [] for k, v in self.channel_index_map.items()},
                **{f"{v}_token_count_list": [] for k, v in self.channel_index_map.items()}
            }
            self._print_debug_info(f"[DEBUG] cumulative_dict init: {self.cumulative_dict}")

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
        assert all_channels is not None, 'No channels found, please check the channel_index_map.'
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
        token_channels = []
        for i in range(bs):
            token_channels.extend([inputs_channels[i]] * seq)
        self.cumulative_dict["local_step_count"] += 1

        for chs in torch.unique(inputs_channels, dim=0):
            indices = [i for i, c in enumerate(token_channels) if (c == chs).all()]
            if not indices:
                continue
            ch_mask = mask[indices]
            ch_losses = token_loss[indices]
            valid_losses = ch_losses[ch_mask]
            total_loss = torch.sum(valid_losses).item()
            token_count = valid_losses.numel()
            for ch in chs:
                self.cumulative_dict[f"{ch}_total_loss_list"].append(total_loss)
                self.cumulative_dict[f"{ch}_token_count_list"].append(token_count)

        # 每gradient_accumulation_steps步聚合
        if self.cumulative_dict["local_step_count"] % self.args.gradient_accumulation_steps == 0:
            for ch in self.all_channels:
                loss_sum = sum(self.cumulative_dict[f"{ch}_total_loss_list"])
                num_items = sum(self.cumulative_dict[f"{ch}_token_count_list"])
                ch_loss = loss_sum / (num_items + 1e-12)
                channel_name = None
                for k, v in self.channel_index_map.items():
                    if v == int(ch):
                        channel_name = k
                        break
                if ch_loss > 0.0:
                    self.cumulative_dict[f"{ch}_ch_steps_count"] += 1
                    checkpoint_step = self.cumulative_dict["local_step_count"] // self.args.gradient_accumulation_steps
                    self.writer.add_scalar(f"train/global_ch_loss_{channel_name}", ch_loss, checkpoint_step)
                    self.writer.add_scalar(f"train/ch_loss_{channel_name}", ch_loss,  self.cumulative_dict[f"{ch}_ch_steps_count"])

            # Reset
            for ch in self.all_channels:
                self.cumulative_dict[f"{ch}_total_loss_list"] = []
                self.cumulative_dict[f"{ch}_token_count_list"] = []

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
