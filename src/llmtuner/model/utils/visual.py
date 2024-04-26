from typing import TYPE_CHECKING, Tuple

import torch

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = get_logger(__name__)


def autocast_projector_dtype(
    model: "PreTrainedModel", model_args: "ModelArguments", mm_projector_name: str = "multi_modal_projector"
) -> None:
    def _mm_projector_forward_post_hook(
        module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        return output.to(model_args.compute_dtype)

    if hasattr(model, mm_projector_name):
        logger.info("Casting multimodal projector outputs in {}.".format(model_args.compute_dtype))
        mm_projector: "torch.nn.Module" = getattr(model, mm_projector_name)
        mm_projector.register_forward_hook(_mm_projector_forward_post_hook)
