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
        self.writer = SummaryWriter(log_dir=self.args.logging_dir, flush_secs=120)   # TODO: 【√】draw channels loss

        if os.path.exists(self.args.logging_dir):
            shutil.rmtree(self.args.logging_dir)
            
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
        if self.finetuning_args.channel_loss:
            self.cumulative_dict = {
                "cumulative_loss": 0.0,
                "accumulated_items": 0,
                "accumulated_steps": 0,
                **{f"{gpu}_{v}_loss": 0.0 for gpu in range(self.gpu_count) for k, v in self.channel_index_map.items()}, # channel 的损失, 粒度为: gpu_channel_loss
                **{f"{gpu}_{v}_count": 0 for gpu in range(torch.cuda.device_count()) for k, v in self.channel_index_map.items()}   # channel 的计数器, 粒度为: gpu_channel_count
            }
            self._print_debug_info(f"[DEBUG] cumulative_dict init: {self.cumulative_dict}")
            # e.g.: self.data_args.channel_loss: {'channel_test_semantic_20240808': 0, 'channel_test_evaluation_good_20240808': 1, 'channel_test_evaluation_general_20240808': 2}
            # e.g.: self.cumulative_dict: {'cumulative_loss': 0.0, 'accumulated_steps': 0, 0: 0.0, 1: 0.0, 2: 0.0, '0_0_count': 0, '0_1_count': 0, '0_2_count': 0, '1_0_count': 0, '1_1_count': 0, '1_2_count': 0}

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

    def training_step(self, model, inputs, num_items_in_batch=None):
        """梯度累积会调用多次"""
        if self.finetuning_args.channel_loss:
            channels = inputs.pop("channel", None)

        loss = super().training_step(model, inputs, num_items_in_batch)

        if channels is not None:
            self.training_step_end(loss, channels, num_items_in_batch)

        return loss

    def training_step_end(self, loss, channels, num_items_in_batch):
        # 累积总损失
        self.cumulative_dict["cumulative_loss"] += loss.item() * num_items_in_batch
        self.cumulative_dict["accumulated_steps"] += 1
        self.cumulative_dict["accumulated_items"] += num_items_in_batch

        # 每个step累积各个 channel 的损失和计数
        for channel in channels:
            curr_gpu = self._get_curr_gpu()
            self.cumulative_dict[f"{curr_gpu}_{channel.item()}_loss"] += loss.item() * num_items_in_batch / len(channels)
            self.cumulative_dict[f"{curr_gpu}_{channel.item()}_count"] += num_items_in_batch / len(channels)

        if self.cumulative_dict["accumulated_steps"] % self.args.gradient_accumulation_steps == 0 and (self.state.global_step+1) % self.args.logging_steps == 0:
            # 每10step 汇聚一次
            if dist.is_initialized():
                # 汇聚总损失
                cumulative_loss_tensor = torch.tensor(self.cumulative_dict["cumulative_loss"]).to('cuda')
                dist.all_reduce(cumulative_loss_tensor, op=dist.ReduceOp.SUM)
                self.cumulative_dict["cumulative_loss"] = cumulative_loss_tensor.item() / dist.get_world_size()

                # # print(f"[DEBUG] GPU: {dist.get_rank()}, dict: {self.cumulative_dict}")

                # 汇聚每个卡的 channel_loss 和 channel_count
                for key, val in self.cumulative_dict.items():
                    if key not in ["cumulative_loss", "accumulated_steps", "accumulated_items"]:
                        loss_tensor = torch.tensor(self.cumulative_dict[key]).to('cuda')
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        self.cumulative_dict[key] = loss_tensor.item()

                # # print(f"[DEBUG 汇聚] GPU: {dist.get_rank()}, dict: {self.cumulative_dict}")

                if dist.get_rank() == 0:
                    tmp_merged_dict = {}
                    for gpu in range(self.gpu_count):
                        for channel_name,channel_id in self.channel_index_map.items():
                            if f"{gpu}_{channel_id}_loss" in self.cumulative_dict and f"{gpu}_{channel_id}_count" in self.cumulative_dict:
                                item_count = self.cumulative_dict[f"{gpu}_{channel_id}_count"]
                                loss = self.cumulative_dict[f"{gpu}_{channel_id}_loss"]
                                if channel_id in tmp_merged_dict:
                                    tmp_merged_dict[channel_id]["loss"] += loss
                                    tmp_merged_dict[channel_id]["count"] += item_count
                                else:
                                    tmp_merged_dict[channel_id] = {"loss":loss,"count":item_count}

                    for key, val in tmp_merged_dict.items():
                        loss_name = [k for k, v in self.channel_index_map.items() if v == int(key)][0]
                        if val["count"] > 0:
                            channel_loss = val["loss"] / val["count"]
                        else:
                            channel_loss = 0.0
                        self.writer.add_scalar(f"train/channel_loss_{loss_name}", channel_loss, self.state.global_step + 1)

                    total_loss = self.cumulative_dict["cumulative_loss"] / self.cumulative_dict["accumulated_items"]
                    self.writer.add_scalar("train/train_loss", total_loss, self.state.global_step + 1)

            else:
                for key, val in self.cumulative_dict.items():
                    if key.endswith('_loss') and key != "cumulative_loss":
                        loss_name = [k for k, v in self.channel_index_map.items() if v == int(key.split("_")[1])][0]
                        channel_loss = val / self.args.logging_steps
                        self.writer.add_scalar(f"train/channel_loss_{loss_name}", channel_loss, self.state.global_step + 1)

                total_loss = self.cumulative_dict["cumulative_loss"] / self.cumulative_dict["accumulated_items"]
                self.writer.add_scalar("train/train_loss", total_loss, self.state.global_step + 1)
            # 10step 重置一次
            self._reset_cumulative()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.finetuning_args.channel_loss:
            _ = inputs.pop("channel", None)

        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

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

    def _reset_cumulative(self):
        """重置累积值
        """
        for key, val in self.cumulative_dict.items():
            if key.endswith('_loss'):
                self.cumulative_dict[key] = 0.0
            else:
                self.cumulative_dict[key] = 0


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
