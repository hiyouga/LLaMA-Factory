from typing import TYPE_CHECKING

from ...extras.constants import MOD_SUPPORTED_MODELS


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


def load_mod_pretrained_model(**init_kwargs) -> "PreTrainedModel":
    from MoD import AutoMoDModelForCausalLM

    return AutoMoDModelForCausalLM.from_pretrained(**init_kwargs)


def convert_pretrained_model_to_mod(
    model: "PreTrainedModel", config: "PretrainedConfig", model_args: "ModelArguments"
) -> "PreTrainedModel":
    from MoD import apply_mod_to_hf

    if getattr(config, "model_type", None) not in MOD_SUPPORTED_MODELS:
        raise ValueError("Current model is not supported by mixture-of-depth.")

    model = apply_mod_to_hf(model)
    model = model.to(model_args.compute_dtype)
    return model
