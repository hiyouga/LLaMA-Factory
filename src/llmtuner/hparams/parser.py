import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_torch_bf16_gpu_available

from ..extras.logging import get_logger
from ..extras.misc import check_dependencies
from ..extras.packages import is_unsloth_available
from .data_args import DataArguments
from .evaluation_args import EvaluationArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments


logger = get_logger(__name__)


check_dependencies()


_TRAIN_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]
_TRAIN_CLS = Tuple[ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]
_INFER_ARGS = [ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_INFER_CLS = Tuple[ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_EVAL_ARGS = [ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]
_EVAL_CLS = Tuple[ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]


def _parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)


def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def _verify_model_args(model_args: "ModelArguments", finetuning_args: "FinetuningArguments") -> None:
    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")

    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantization is only compatible with the LoRA method.")

        if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create new adapter upon a quantized model.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("Quantized model only accepts a single adapter. Merge them first.")


def _parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return _parse_args(parser, args)


def _parse_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    return _parse_args(parser, args)


def _parse_eval_args(args: Optional[Dict[str, Any]] = None) -> _EVAL_CLS:
    parser = HfArgumentParser(_EVAL_ARGS)
    return _parse_args(parser, args)


def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    # Check arguments
    if finetuning_args.stage != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if finetuning_args.stage != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if finetuning_args.stage == "ppo" and not training_args.do_train:
        raise ValueError("PPO training does not support evaluation, use the SFT stage to evaluate models.")

    if finetuning_args.stage == "ppo" and model_args.shift_attn:
        raise ValueError("PPO training is incompatible with S^2-Attn.")

    if finetuning_args.stage == "ppo" and finetuning_args.reward_model_type == "lora" and model_args.use_unsloth:
        raise ValueError("Unsloth does not support lora reward model.")

    if (
        finetuning_args.stage == "ppo"
        and training_args.report_to is not None
        and training_args.report_to[0] not in ["wandb", "tensorboard"]
    ):
        raise ValueError("PPO only accepts wandb or tensorboard logger.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True while training.")

    if training_args.do_train and model_args.use_unsloth and not is_unsloth_available():
        raise ValueError("Unsloth was not installed: https://github.com/unslothai/unsloth")

    if finetuning_args.use_dora and model_args.use_unsloth:
        raise ValueError("Unsloth does not support DoRA.")

    if finetuning_args.pure_bf16:
        if not is_torch_bf16_gpu_available():
            raise ValueError("This device does not support `pure_bf16`.")

        if training_args.fp16 or training_args.bf16:
            raise ValueError("Turn off mixed precision training when using `pure_bf16`.")

    if (
        finetuning_args.use_galore
        and finetuning_args.galore_layerwise
        and training_args.parallel_mode.value == "distributed"
    ):
        raise ValueError("Distributed training does not support layer-wise GaLore.")

    if finetuning_args.use_galore and training_args.deepspeed is not None:
        raise ValueError("GaLore is incompatible with DeepSpeed.")

    if model_args.infer_backend == "vllm":
        raise ValueError("vLLM backend is only available for API, CLI and Web.")

    _verify_model_args(model_args, finetuning_args)

    if (
        training_args.do_train
        and finetuning_args.finetuning_type == "lora"
        and model_args.resize_vocab
        and finetuning_args.additional_target is None
    ):
        logger.warning("Add token embeddings to `additional_target` to make the added tokens trainable.")

    if training_args.do_train and model_args.quantization_bit is not None and (not model_args.upcast_layernorm):
        logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training.")

    if training_args.do_train and finetuning_args.use_galore and not finetuning_args.pure_bf16:
        logger.warning("Using GaLore with mixed precision training may significantly increases GPU memory usage.")

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (not training_args.do_train) and finetuning_args.stage == "dpo" and finetuning_args.ref_model is None:
        logger.warning("Specify `ref_model` for computing rewards at evaluation.")

    # Post-process training arguments
    if (
        training_args.parallel_mode.value == "distributed"
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args.ddp_find_unused_parameters = False

    if finetuning_args.stage in ["rm", "ppo"] and finetuning_args.finetuning_type in ["full", "freeze"]:
        can_resume_from_checkpoint = False
        if training_args.resume_from_checkpoint is not None:
            logger.warning("Cannot resume from checkpoint in current stage.")
            training_args.resume_from_checkpoint = None
    else:
        can_resume_from_checkpoint = True

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
        and can_resume_from_checkpoint
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info(
                "Resuming training from {}. Change `output_dir` or use `overwrite_output_dir` to avoid.".format(
                    training_args.resume_from_checkpoint
                )
            )

    if (
        finetuning_args.stage in ["rm", "ppo"]
        and finetuning_args.finetuning_type == "lora"
        and training_args.resume_from_checkpoint is not None
    ):
        logger.warning(
            "Add {} to `adapter_name_or_path` to resume training from checkpoint.".format(
                training_args.resume_from_checkpoint
            )
        )

    # Post-process model arguments
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    model_args.model_max_length = data_args.cutoff_len
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    # Log on each process the small summary:
    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            training_args.parallel_mode.value == "distributed",
            str(model_args.compute_dtype),
        )
    )

    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args


def get_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    model_args, data_args, finetuning_args, generating_args = _parse_infer_args(args)

    _set_transformers_logging()

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if model_args.infer_backend == "vllm":
        if finetuning_args.stage != "sft":
            raise ValueError("vLLM engine only supports auto-regressive models.")

        if model_args.adapter_name_or_path is not None:
            raise ValueError("vLLM engine does not support LoRA adapters. Merge them first.")

        if model_args.quantization_bit is not None:
            raise ValueError("vLLM engine does not support quantization.")

        if model_args.rope_scaling is not None:
            raise ValueError("vLLM engine does not support RoPE scaling.")

    _verify_model_args(model_args, finetuning_args)

    model_args.device_map = "auto"

    return model_args, data_args, finetuning_args, generating_args


def get_eval_args(args: Optional[Dict[str, Any]] = None) -> _EVAL_CLS:
    model_args, data_args, eval_args, finetuning_args = _parse_eval_args(args)

    _set_transformers_logging()

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if model_args.infer_backend == "vllm":
        raise ValueError("vLLM backend is only available for API, CLI and Web.")

    _verify_model_args(model_args, finetuning_args)

    model_args.device_map = "auto"

    transformers.set_seed(eval_args.seed)

    return model_args, data_args, eval_args, finetuning_args
