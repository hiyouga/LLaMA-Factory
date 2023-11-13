import transformers
from typing import Any, Dict, Optional, Tuple
from transformers import HfArgumentParser

from llmtuner.extras.misc import parse_args
from llmtuner.hparams import (
    ModelArguments,
    DataArguments,
    EvaluationArguments,
    FinetuningArguments
)


def parse_eval_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    EvaluationArguments,
    FinetuningArguments
]:
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        EvaluationArguments,
        FinetuningArguments
    ))
    return parse_args(parser, args)


def get_eval_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    EvaluationArguments,
    FinetuningArguments
]:
    model_args, data_args, eval_args, finetuning_args = parse_eval_args(args)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    transformers.set_seed(eval_args.seed)

    return model_args, data_args, eval_args, finetuning_args
