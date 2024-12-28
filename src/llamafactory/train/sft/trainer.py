# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


# *************** AIFactory Custom Code Begin ***************
import time
import math
from transformers.utils import (
    is_torch_xla_available,
)
from transformers.trainer_utils import (
    speed_metrics
)
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
else:
    IS_XLA_FSDPV2_POST_2_2 = False

from transformers.debug_utils import DebugOption
from torch.utils.data import Dataset
# *************** AIFactory Custom Code End ***************


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

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
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            logger.info(f'padding_side: {self.tokenizer.padding_side}')
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))

    
# *************** AIFactory Custom Code Begin ***************
    # def _merge_eval_metrics(self, metrics: Dict[str, float], eval_dataset: Dict[str, Dataset]) -> Dict[str, float]:
    #     # 将metrics按照每个数据集的样本数进行加权平均
    #     new_metrics = {}
    #     all_losses = []
    #     all_runtime = []
    #     all_samples_per_second = []
    #     all_steps_per_second = []
    #     all_epochs = []
    #     for dataset_name, dataset in eval_dataset.items():
    #         loss = metrics[f'eval_{dataset_name}_loss']
    #         all_losses.append(loss * len(dataset))
    #         new_metrics[f'eval_{dataset_name}_loss'] = loss
            
    #         runtime = metrics[f'eval_{dataset_name}_runtime']
    #         all_runtime.append(runtime * len(dataset))

    #         samples_per_second = metrics[f'eval_{dataset_name}_samples_per_second']
    #         all_samples_per_second.append(samples_per_second * len(dataset))

    #         steps_per_second = metrics[f'eval_{dataset_name}_steps_per_second']
    #         all_steps_per_second.append(steps_per_second * len(dataset))

    #         epoch = metrics[f'epoch']
    #         all_epochs.append(epoch * len(dataset))

    #     new_metrics['eval_loss'] = sum(all_losses) / sum(len(dataset) for dataset in eval_dataset.values())
    #     new_metrics['eval_runtime'] = sum(all_runtime) / sum(len(dataset) for dataset in eval_dataset.values())
    #     new_metrics['eval_samples_per_second'] = sum(all_samples_per_second) / sum(len(dataset) for dataset in eval_dataset.values())
    #     new_metrics['eval_steps_per_second'] = sum(all_steps_per_second) / sum(len(dataset) for dataset in eval_dataset.values())
    #     new_metrics['epoch'] = sum(all_epochs) / sum(len(dataset) for dataset in eval_dataset.values())

    #     return new_metrics


    # def aif_evaluate(
    #     self,
    #     eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    #     **gen_kwargs,
    # ) -> Dict[str, float]:
    #     """
    #     Run evaluation and returns metrics.

    #     The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
    #     (pass it to the init `compute_metrics` argument).

    #     You can also subclass and override this method to inject custom behavior.

    #     Args:
    #         eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
    #             Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
    #             not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
    #             evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
    #             `__len__` method.

    #             <Tip>

    #             If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
    #             separate evaluations on each dataset. This can be useful to monitor how training affects other
    #             datasets or simply to get a more fine-grained evaluation.
    #             When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
    #             of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
    #             `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
    #             loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

    #             </Tip>

    #         ignore_keys (`List[str]`, *optional*):
    #             A list of keys in the output of your model (if it is a dictionary) that should be ignored when
    #             gathering predictions.
    #         metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
    #             An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
    #             "eval_bleu" if the prefix is "eval" (default)

    #     Returns:
    #         A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
    #         dictionary also contains the epoch number which comes from the training state.
    #     """
    #     gen_kwargs = gen_kwargs.copy()
    #     print(f'padding_side: {self.tokenizer.padding_side}')

    #     # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
    #     # training args
    #     if (
    #         gen_kwargs.get("max_length") is None
    #         and gen_kwargs.get("max_new_tokens") is None
    #         and self.args.generation_max_length is not None
    #     ):
    #         gen_kwargs["max_length"] = self.args.generation_max_length
    #     if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
    #         gen_kwargs["num_beams"] = self.args.generation_num_beams
    #     # We don't want to drop samples in general
    #     self.gather_function = self.accelerator.gather
    #     self._gen_kwargs = gen_kwargs

    #     # handle multipe eval datasets
    #     override = eval_dataset is not None
    #     eval_dataset = eval_dataset if override else self.eval_dataset
    #     if isinstance(eval_dataset, dict):
    #         metrics = {}
    #         for eval_dataset_name, _eval_dataset in eval_dataset.items():
    #             dataset_metrics = self.evaluate(
    #                 eval_dataset=_eval_dataset if override else eval_dataset_name,
    #                 ignore_keys=ignore_keys,
    #                 metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
    #             )
    #             metrics.update(dataset_metrics)
    #         # 将metrics按照每个数据集的样本数进行加权平均
    #         metrics = self._merge_eval_metrics(metrics, eval_dataset)
    #         return metrics

    #     # memory metrics - must set up as early as possible
    #     self._memory_tracker.start()

    #     eval_dataloader = self.get_eval_dataloader(eval_dataset)
    #     if self.is_fsdp_xla_v2_enabled:
    #         eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

    #     start_time = time.time()

    #     eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
    #     output = eval_loop(
    #         eval_dataloader,
    #         description="Evaluation",
    #         # No point gathering the predictions if there are no metrics, otherwise we defer to
    #         # self.args.prediction_loss_only
    #         prediction_loss_only=True if self.compute_metrics is None else None,
    #         ignore_keys=ignore_keys,
    #         metric_key_prefix=metric_key_prefix,
    #     )

    #     total_batch_size = self.args.eval_batch_size * self.args.world_size
    #     if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
    #         start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
    #     if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
    #         start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
    #     output.metrics.update(
    #         speed_metrics(
    #             metric_key_prefix,
    #             start_time,
    #             num_samples=output.num_samples,
    #             num_steps=math.ceil(output.num_samples / total_batch_size),
    #         )
    #     )

    #     self.log(output.metrics)

    #     if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #         # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #         xm.master_print(met.metrics_report())

    #     self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

    #     self._memory_tracker.stop_and_update_metrics(output.metrics)

    #     return output.metrics
# *************** AIFactory Custom Code End ***************