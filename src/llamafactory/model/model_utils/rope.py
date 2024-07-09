import math
from typing import TYPE_CHECKING

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = get_logger(__name__)


def configure_rope(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if model_args.rope_scaling is None:
        return

    if not hasattr(config, "rope_scaling"):
        logger.warning("Current model does not support RoPE scaling.")
        return

    if is_trainable:
        if model_args.rope_scaling == "dynamic":
            logger.warning(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        current_max_length = getattr(config, "max_position_embeddings", None)
        if current_max_length and model_args.model_max_length > current_max_length:
            logger.info(
                "Enlarge max model length from {} to {}.".format(current_max_length, model_args.model_max_length)
            )
            setattr(config, "max_position_embeddings", model_args.model_max_length)
            scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))
        else:
            logger.warning("Input length is smaller than max length. Consider increase input length.")
            scaling_factor = 1.0
    else:
        scaling_factor = 2.0

    setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
    logger.info(
        "Using {} scaling strategy and setting scaling factor to {}".format(model_args.rope_scaling, scaling_factor)
    )
