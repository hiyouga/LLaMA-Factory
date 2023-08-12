import os
import sys
import torch
import datasets
import transformers
from typing import Any, Dict, Optional, Tuple
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from llmtuner.extras.logging import get_logger
from llmtuner.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    GeneralArguments
)


logger = get_logger(__name__)


def _parse_args(parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()


def parse_train_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    Seq2SeqTrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
    GeneralArguments
]:
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        Seq2SeqTrainingArguments,
        FinetuningArguments,
        GeneratingArguments,
        GeneralArguments
    ))
    return _parse_args(parser, args)


def parse_infer_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments
]:
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        FinetuningArguments,
        GeneratingArguments
    ))
    return _parse_args(parser, args)


def get_train_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    Seq2SeqTrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
    GeneralArguments
]:
    model_args, data_args, training_args, finetuning_args, generating_args, general_args = parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Check arguments (do not check finetuning_args since it may be loaded from checkpoints)
    data_args.init_for_training()

    if general_args.stage != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True while training.")

    if general_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if general_args.stage in ["rm", "ppo"] and finetuning_args.finetuning_type != "lora":
        raise ValueError("RM and PPO training can only be performed with the LoRA method.")

    if general_args.stage in ["ppo", "dpo"] and not training_args.do_train:
        raise ValueError("PPO and DPO stage can only be performed at training.")

    if general_args.stage == "ppo" and model_args.reward_model is None:
        raise ValueError("Reward model is necessary for PPO training.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if general_args.stage == "ppo" and data_args.streaming:
        raise ValueError("Streaming mode does not suppport PPO training currently.")

    if data_args.val_size > 1e-6 and data_args.val_size < 1 and data_args.streaming:
        raise ValueError("Streaming mode should have an integer val size.")

    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora":
            if len(model_args.checkpoint_dir) != 1:
                raise ValueError("Only LoRA tuning accepts multiple checkpoints.")
        elif model_args.quantization_bit is not None and len(model_args.checkpoint_dir) != 1:
                raise ValueError("Quantized model only accepts a single checkpoint.")

    if model_args.quantization_bit is not None and (not training_args.do_train):
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if training_args.do_train and (not training_args.fp16):
        logger.warning("We recommend enable fp16 mixed precision training.")

    if (
        training_args.local_rank != -1
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args.ddp_find_unused_parameters = False

    if data_args.max_samples is not None and data_args.streaming:
        logger.warning("`max_samples` is incompatible with `streaming`. Disabling max_samples.")
        data_args.max_samples = None

    training_args.optim = "adamw_torch" if training_args.optim == "adamw_hf" else training_args.optim # suppress warning

    if training_args.fp16:
        model_args.compute_dtype = torch.float16
    elif training_args.bf16:
        if not torch.cuda.is_bf16_supported():
            raise ValueError("Current device does not support bf16 training.")
        model_args.compute_dtype = torch.bfloat16
    else:
        model_args.compute_dtype = torch.float32

    model_args.model_max_length = data_args.max_source_length + data_args.max_target_length

    # Log on each process the small summary:
    logger.info("Process rank: {}, device: {}, n_gpu: {}\n  distributed training: {}, 16-bits training: {}".format(
        training_args.local_rank, training_args.device, training_args.n_gpu,
        bool(training_args.local_rank != -1), training_args.fp16
    ))
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args, general_args


def get_infer_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments
]:
    model_args, data_args, finetuning_args, generating_args = parse_infer_args(args)

    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora":
            if len(model_args.checkpoint_dir) != 1:
                raise ValueError("Only LoRA tuning accepts multiple checkpoints.")
        elif model_args.quantization_bit is not None and len(model_args.checkpoint_dir) != 1:
                raise ValueError("Quantized model only accepts a single checkpoint.")

    return model_args, data_args, finetuning_args, generating_args
