from typing import TYPE_CHECKING

from ...extras.logging import get_logger
from ...extras.packages import is_flash_attn2_available, is_sdpa_available


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = get_logger(__name__)


def configure_attn_implementation(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    if model_args.flash_attn == "auto":
        return

    elif model_args.flash_attn == "off":
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == "sdpa":
        if not is_sdpa_available():
            logger.warning("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == "fa2":
        if not is_flash_attn2_available():
            logger.warning("FlashAttention-2 is not installed.")
            return

        requested_attn_implementation = "flash_attention_2"
    else:
        raise NotImplementedError("Unknown attention type: {}".format(model_args.flash_attn))

    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info("Using torch SDPA for faster training and inference.")
    else:
        logger.info("Using vanilla attention implementation.")
