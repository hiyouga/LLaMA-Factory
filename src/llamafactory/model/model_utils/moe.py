# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Union

from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras.misc import check_version


if TYPE_CHECKING:
    from torch import nn
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


def _set_z3_leaf_modules(model: "PreTrainedModel", leaf_modules: list[Union["nn.Module", str]]) -> None:
    check_version("deepspeed>=0.13.0")
    from deepspeed.utils import set_z3_leaf_modules  # type: ignore

    set_z3_leaf_modules(model, leaf_modules)


def add_z3_leaf_module(model: "PreTrainedModel") -> None:
    r"""Set module as a leaf module to skip partitioning in deepspeed zero3."""
    if not is_deepspeed_zero3_enabled():
        return

    model_type = getattr(model.config, "model_type", None)
    if model_type == "dbrx":
        from transformers.models.dbrx.modeling_dbrx import DbrxFFN

        _set_z3_leaf_modules(model, [DbrxFFN])

    if model_type == "deepseek_v2":
        # deepseek v2 uses custom code
        _set_z3_leaf_modules(model, ["DeepseekV2MoE"])

    if model_type == "deepseek_v3" or model_type == "kimi_vl":
        # deepseek v3 and kimi vl use custom code
        _set_z3_leaf_modules(model, ["DeepseekV3MoE"])

    if model_type == "granitemoe":
        from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE

        _set_z3_leaf_modules(model, [GraniteMoeMoE])

    if model_type == "glm4_moe":
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE

        _set_z3_leaf_modules(model, [Glm4MoeMoE])

    if model_type == "jamba":
        from transformers.models.jamba.modeling_jamba import JambaSparseMoeBlock

        _set_z3_leaf_modules(model, [JambaSparseMoeBlock])

    if model_type == "jetmoe":
        from transformers.models.jetmoe.modeling_jetmoe import JetMoeMoA, JetMoeMoE

        _set_z3_leaf_modules(model, [JetMoeMoA, JetMoeMoE])

    if model_type == "llama4":
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

        _set_z3_leaf_modules(model, [Llama4TextMoe])

    if model_type == "mixtral":
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        _set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if model_type == "olmoe":
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

        _set_z3_leaf_modules(model, [OlmoeSparseMoeBlock])

    if model_type == "phimoe":
        from transformers.models.phimoe.modeling_phimoe import PhimoeSparseMoeBlock

        _set_z3_leaf_modules(model, [PhimoeSparseMoeBlock])

    if model_type == "qwen2_moe":
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

        _set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])

    if model_type == "qwen3_moe":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        _set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])


def configure_moe(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.moe_aux_loss_coef:
        return

    model_type = getattr(config, "model_type", None)
    if model_type in [
        "dbrx",
        "granitemoe",
        "jamba",
        "jetmoe",
        "llama4",
        "mixtral",
        "olmoe",
        "phimoe",
        "qwen2_moe",
        "qwen3_moe",
    ]:
        setattr(config, "output_router_logits", True)

    if model_type in ["granitemoe", "jamba", "llama4", "mixtral", "olmoe", "phimoe", "qwen2_moe", "qwen3_moe"]:
        setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)

    elif model_type == "deepseek":
        setattr(config, "aux_loss_alpha", model_args.moe_aux_loss_coef)

    elif model_type == "jetmoe":
        setattr(config, "aux_loss_coef", model_args.moe_aux_loss_coef)
