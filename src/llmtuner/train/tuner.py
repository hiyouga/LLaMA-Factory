from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..extras.callbacks import LogCallback
from ..extras.logging import get_logger
from ..hparams import get_infer_args, get_train_args
from ..model import load_model_and_tokenizer
from .dpo import run_dpo
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError("Unknown task.")


def export_model(args: Optional[Dict[str, Any]] = None):
    model_args, _, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir`.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)

    if getattr(model, "quantization_method", None) and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None):
        model = model.to("cpu")
    elif hasattr(model.config, "torch_dtype"):
        model = model.to(getattr(model.config, "torch_dtype")).to("cpu")
    else:
        model = model.to(torch.float16).to("cpu")
        setattr(model.config, "torch_dtype", torch.float16)

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size="{}GB".format(model_args.export_size),
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size="{}GB".format(model_args.export_size),
            safe_serialization=(not model_args.export_legacy_format),
        )

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)
    except Exception:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    run_exp()
