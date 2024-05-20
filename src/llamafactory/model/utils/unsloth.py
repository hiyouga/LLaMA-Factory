from typing import TYPE_CHECKING, Any, Dict, Optional

from ...extras.logging import get_logger
from ...extras.misc import get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


logger = get_logger(__name__)


def _get_unsloth_kwargs(
    config: "PretrainedConfig", model_name_or_path: str, model_args: "ModelArguments"
) -> Dict[str, Any]:
    return {
        "model_name": model_name_or_path,
        "max_seq_length": model_args.model_max_length or 4096,
        "dtype": model_args.compute_dtype,
        "load_in_4bit": model_args.quantization_bit == 4,
        "token": model_args.hf_hub_token,
        "device_map": {"": get_current_device()},
        "rope_scaling": getattr(config, "rope_scaling", None),
        "fix_tokenizer": False,
        "trust_remote_code": True,
        "use_gradient_checkpointing": "unsloth",
    }


def load_unsloth_pretrained_model(
    config: "PretrainedConfig", model_args: "ModelArguments"
) -> Optional["PreTrainedModel"]:
    r"""
    Optionally loads pretrained model with unsloth. Used in training.
    """
    from unsloth import FastLanguageModel

    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.model_name_or_path, model_args)
    try:
        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        logger.warning("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
        model = None
        model_args.use_unsloth = False

    return model


def get_unsloth_peft_model(
    model: "PreTrainedModel", model_args: "ModelArguments", peft_kwargs: Dict[str, Any]
) -> "PreTrainedModel":
    r"""
    Gets the peft model for the pretrained model with unsloth. Used in training.
    """
    from unsloth import FastLanguageModel

    unsloth_peft_kwargs = {
        "model": model,
        "max_seq_length": model_args.model_max_length,
        "use_gradient_checkpointing": "unsloth",
    }
    return FastLanguageModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)


def load_unsloth_peft_model(
    config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool
) -> "PreTrainedModel":
    r"""
    Loads peft model with unsloth. Used in both training and inference.
    """
    from unsloth import FastLanguageModel

    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.adapter_name_or_path[0], model_args)
    try:
        if not is_trainable:
            unsloth_kwargs["use_gradient_checkpointing"] = False

        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        raise ValueError("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))

    if not is_trainable:
        FastLanguageModel.for_inference(model)

    return model
