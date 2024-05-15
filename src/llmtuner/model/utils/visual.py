from typing import TYPE_CHECKING, Tuple

import torch
import transformers
from torch import nn

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, LlavaConfig

    from ...hparams import ModelArguments


logger = get_logger(__name__)


def configure_hidden_size(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "llava":
        setattr(config, "hidden_size", getattr(config.text_config, "hidden_size", None))


def autocast_projector_dtype(
    model: "PreTrainedModel", model_args: "ModelArguments", mm_projector_name: str = "multi_modal_projector"
) -> None:
    def _mm_projector_forward_post_hook(
        module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        return output.to(model_args.compute_dtype)

    if hasattr(model, mm_projector_name) and (getattr(model.config, "quantization_method", None)
                                              or "Yi" in getattr(model.config.text_config, "_name_or_path", None)):
        logger.info("Casting multimodal projector outputs in {}.".format(model_args.compute_dtype))
        mm_projector: "torch.nn.Module" = getattr(model, mm_projector_name)
        mm_projector.register_forward_hook(_mm_projector_forward_post_hook)


class LlavaMultiModalProjectorYiVL(nn.Module):
    def __init__(self, config: "LlavaConfig"):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_2 = nn.LayerNorm(config.text_config.hidden_size, bias=True)
        self.linear_3 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_4 = nn.LayerNorm(config.text_config.hidden_size, bias=True)
        self.act = nn.GELU()

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        hidden_states = self.linear_4(hidden_states)
        return hidden_states


def configure_visual(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    logger = get_logger(__name__)
    if model_args.visual_inputs and "Yi" in getattr(config.text_config, "_name_or_path", None):
        transformers.models.llava.modeling_llava.LlavaMultiModalProjector = LlavaMultiModalProjectorYiVL
        logger.info("Patched Multimodal Projector for Yi-VL.")
