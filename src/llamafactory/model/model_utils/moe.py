from typing import TYPE_CHECKING

from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


def add_z3_leaf_module(model: "PreTrainedModel") -> None:
    r"""
    Sets module as a leaf module to skip partitioning in deepspeed zero3.
    """
    if not is_deepspeed_zero3_enabled():
        return

    require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
    from deepspeed.utils import set_z3_leaf_modules  # type: ignore

    if getattr(model.config, "model_type", None) == "dbrx":
        from transformers.models.dbrx.modeling_dbrx import DbrxFFN

        set_z3_leaf_modules(model, [DbrxFFN])

    if getattr(model.config, "model_type", None) == "jamba":
        from transformers.models.jamba.modeling_jamba import JambaSparseMoeBlock

        set_z3_leaf_modules(model, [JambaSparseMoeBlock])

    if getattr(model.config, "model_type", None) == "jetmoe":
        from transformers.models.jetmoe.modeling_jetmoe import JetMoeMoA, JetMoeMoE

        set_z3_leaf_modules(model, [JetMoeMoA, JetMoeMoE])

    if getattr(model.config, "model_type", None) == "mixtral":
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if getattr(model.config, "model_type", None) == "qwen2moe":
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

        set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])


def configure_moe(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if model_args.moe_aux_loss_coef is not None:
        if getattr(config, "model_type", None) in ["jamba", "mixtral", "qwen2_moe"]:
            setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)

        elif getattr(config, "model_type", None) == "deepseek":
            setattr(config, "aux_loss_alpha", model_args.moe_aux_loss_coef)

        elif getattr(config, "model_type", None) == "jetmoe":
            setattr(config, "aux_loss_coef", model_args.moe_aux_loss_coef)

    if getattr(config, "model_type", None) in ["dbrx", "jamba", "jetmoe", "mixtral", "qwen2_moe"]:
        setattr(config, "output_router_logits", is_trainable)
