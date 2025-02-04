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

from typing import TYPE_CHECKING, Sequence

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras.misc import check_version


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


def _set_z3_leaf_modules(model: "PreTrainedModel", leaf_modules: Sequence["torch.nn.Module"]) -> None:
    check_version("deepspeed>=0.13.0")
    from deepspeed.utils import set_z3_leaf_modules  # type: ignore

    set_z3_leaf_modules(model, leaf_modules)


def add_z3_leaf_module(model: "PreTrainedModel") -> None:
    r"""
    Sets module as a leaf module to skip partitioning in deepspeed zero3.
    """
    if not is_deepspeed_zero3_enabled():
        return

    model_type = getattr(model.config, "model_type", None)
    if model_type == "dbrx":
        from transformers.models.dbrx.modeling_dbrx import DbrxFFN

        _set_z3_leaf_modules(model, [DbrxFFN])

    if model_type == "jamba":
        from transformers.models.jamba.modeling_jamba import JambaSparseMoeBlock

        _set_z3_leaf_modules(model, [JambaSparseMoeBlock])

    if model_type == "jetmoe":
        from transformers.models.jetmoe.modeling_jetmoe import JetMoeMoA, JetMoeMoE

        _set_z3_leaf_modules(model, [JetMoeMoA, JetMoeMoE])

    if model_type == "mixtral":
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        _set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if model_type == "qwen2_moe":
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

        _set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])


def configure_moe(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    model_type = getattr(config, "model_type", None)
    if model_args.moe_aux_loss_coef is not None:
        if model_type in ["jamba", "mixtral", "qwen2_moe"]:
            setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)

        elif model_type == "deepseek":
            setattr(config, "aux_loss_alpha", model_args.moe_aux_loss_coef)

        elif model_type == "jetmoe":
            setattr(config, "aux_loss_coef", model_args.moe_aux_loss_coef)

    if model_type in ["dbrx", "jamba", "jetmoe", "mixtral", "qwen2_moe"]:
        setattr(config, "output_router_logits", is_trainable)
