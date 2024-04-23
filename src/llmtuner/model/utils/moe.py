from typing import TYPE_CHECKING

from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version


if TYPE_CHECKING:
    from transformers import PreTrainedModel


def add_z3_leaf_module(model: "PreTrainedModel") -> None:
    r"""
    Sets module as a leaf module to skip partitioning in deepspeed zero3.
    """
    if not is_deepspeed_zero3_enabled():
        return

    require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
    from deepspeed.utils import set_z3_leaf_modules  # type: ignore

    if getattr(model.config, "model_type", None) == "mixtral":
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if getattr(model.config, "model_type", None) == "qwen2moe":
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

        set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])

    if getattr(model.config, "model_type", None) == "jamba":
        from transformers.models.jamba.modeling_jamba import JambaSparseMoeBlock

        set_z3_leaf_modules(model, [JambaSparseMoeBlock])

    if getattr(model.config, "model_type", None) == "dbrx":
        from transformers.models.dbrx.modeling_dbrx import DbrxFFN

        set_z3_leaf_modules(model, [DbrxFFN])
