# Copyright 2025 the ROLL team and the LlamaFactory team.
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
"""MCA (mcore_adapter) workflows for PT/SFT/DPO stages, aligned with LLaMA-Factory's workflow style."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from ...data import (
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from ...data.collator import (
    PairwiseDataCollatorWithPadding,
)
from ...extras.constants import IGNORE_INDEX, MCA_SUPPORTED_MODELS
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_mcore_adapter_available
from ...extras.ploting import plot_loss
from ...model import load_tokenizer
from ..callbacks import SaveProcessorCallback


if not is_mcore_adapter_available():
    raise ImportError("mcore_adapter is not installed. Please install it with `pip install mcore-adapter`.")

from mcore_adapter.models import AutoConfig, AutoModel
from mcore_adapter.trainer import DPOTrainer as McaDPOTrainer
from mcore_adapter.trainer import McaTrainer
from mcore_adapter.trainer.dpo_config import DPOConfig
from mcore_adapter.training_args import Seq2SeqTrainingArguments as McaSeq2SeqTrainingArguments


if TYPE_CHECKING:
    from transformers import DataCollatorForSeq2Seq, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def _data_collator_wrapper(data_collator: Any):
    @functools.wraps(data_collator)
    def wrapper(features: Sequence[dict[str, Any]]):
        labels_key = [k for k in features[0].keys() if k.endswith("labels")]
        input_ids_key = [k for k in features[0].keys() if k.endswith("input_ids")]
        for feature in features:
            if len(labels_key) == 0:  # pt
                feature["labels"] = deepcopy(feature["input_ids"])[1:]
            for k in labels_key:
                feature[k] = feature[k][1:]
            for k in input_ids_key:
                feature[k] = feature[k][:-1]
            for k in ["attention_mask", "position_ids"]:
                if k in feature:
                    feature[k] = feature[k][:-1]
        return data_collator(features)

    return wrapper

def _check_model_support(model_args: ModelArguments):
    from transformers import AutoConfig as HfAutoConfig
    config = HfAutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
    if config.model_type not in MCA_SUPPORTED_MODELS:
        raise ValueError(f"Model {config.model_type} is not supported by MCA.")

def run_pt(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: McaSeq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    callbacks: list[TrainerCallback] | None = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # dataset needs +1 then cut back due to MCA shift logic
    data_args.cutoff_len += 1
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)
    data_args.cutoff_len -= 1

    _check_model_support(model_args)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)

    from transformers import DataCollatorForSeq2Seq

    data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )
    data_collator = _data_collator_wrapper(data_collator)

    trainer = McaTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
    )

    if "processor" in tokenizer_module and tokenizer_module["processor"] is not None:
        trainer.add_callback(SaveProcessorCallback(tokenizer_module["processor"]))

    if training_args.do_train:
        train_result = trainer.train(training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]
            plot_loss(training_args.output_dir, keys=keys)


def run_sft(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: McaSeq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    callbacks: list[TrainerCallback] | None = None,
):
    # align packing flags
    # TODO: FIX SequencePacking
    data_args.neat_packing = training_args.sequence_packing = data_args.neat_packing or training_args.sequence_packing
    data_args.packing = data_args.neat_packing or data_args.packing

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # dataset needs +1 then cut back due to MCA shift logic
    data_args.cutoff_len += 1
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    data_args.cutoff_len -= 1

    _check_model_support(model_args)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)

    # optional freezing for qwen2_vl, qwen2_5_vl
    if getattr(model.config, "hf_model_type", None) in ["qwen2_vl", "qwen2_5_vl"] and finetuning_args.freeze_vision_tower:
        for name, p in model.named_parameters():
            if any(name.startswith(k) for k in ["vision_model.blocks", "vision_model.patch_embed"]):
                p.requires_grad_(False)
    if getattr(model.config, "hf_model_type", None) in ["qwen2_vl", "qwen2_5_vl"] and finetuning_args.freeze_multi_modal_projector:
        for name, p in model.named_parameters():
            if any(name.startswith(k) for k in ["multi_modal_projector"]):
                p.requires_grad_(False)
    if getattr(model.config, "hf_model_type", None) in ["qwen2_vl", "qwen2_5_vl"] and finetuning_args.freeze_language_model:
        for name, p in model.named_parameters():
            if any(name.startswith(k) for k in ["embedding", "decoder", "output_layer"]):
                p.requires_grad_(False)

    pad_to_max = (
        training_args.expert_model_parallel_size is not None and training_args.expert_model_parallel_size > 1
    )
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        padding="max_length" if pad_to_max else "longest",
        max_length=data_args.cutoff_len if pad_to_max else None,
        pad_to_multiple_of=64,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    data_collator = _data_collator_wrapper(data_collator)

    trainer = McaTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
    )

    if "processor" in tokenizer_module and tokenizer_module["processor"] is not None:
        trainer.add_callback(SaveProcessorCallback(tokenizer_module["processor"]))

    train_result = trainer.train(training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        keys = ["loss"]
        if isinstance(dataset_module.get("eval_dataset"), dict):
            keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
        else:
            keys += ["eval_loss"]
        plot_loss(training_args.output_dir, keys=keys)


def run_dpo(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: McaSeq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    callbacks: list[TrainerCallback] | None = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    _check_model_support(model_args)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)

    if finetuning_args.use_ref_model:
        ref_config = AutoConfig.from_pretrained(model_args.model_name_or_path, training_args)
        ref_model = AutoModel.from_config(ref_config)
        ref_model.load_state_dict(model.state_dict())
    else:
        ref_model = None

    # dataset needs +1 then cut back due to MCA shift logic
    data_args.cutoff_len += 1
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    data_args.cutoff_len -= 1

    pad_to_max = (
        training_args.expert_model_parallel_size is not None and training_args.expert_model_parallel_size > 1
    )
    dpo_config = DPOConfig(
        beta=finetuning_args.pref_beta,
        pref_loss=finetuning_args.pref_loss,
        label_smoothing=finetuning_args.dpo_label_smoothing,
    )
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        pad_to_multiple_of=64,
        padding="max_length" if pad_to_max else "longest",
        max_length=data_args.cutoff_len if pad_to_max else None,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    data_collator = _data_collator_wrapper(data_collator)

    trainer = McaDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_config=dpo_config,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
    )

    if "processor" in tokenizer_module and tokenizer_module["processor"] is not None:
        trainer.add_callback(SaveProcessorCallback(tokenizer_module["processor"]))

    train_result = trainer.train(training_args.resume_from_checkpoint)
    trainer.save_model()
    if finetuning_args.include_effective_tokens_per_second:
        train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
            dataset_module["train_dataset"], train_result.metrics, stage="rm"
        )

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        keys = ["loss", "rewards/accuracies"]
        if isinstance(dataset_module.get("eval_dataset"), dict):
            keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
        else:
            keys += ["eval_loss"]

        plot_loss(training_args.output_dir, keys=keys)

