# Copyright 2024 the LlamaFactory team.
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

import json
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras.packages import is_requests_available


if is_requests_available():
    import requests


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead


def get_rewards_from_server(server_url: str, messages: List[str]) -> List["torch.Tensor"]:
    r"""
    Gets reward scores from the API server.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(server_url, json=payload, headers=headers)
    rewards = json.loads(response.text)["scores"]
    return torch.Tensor(rewards)


def replace_model(model: "AutoModelForCausalLMWithValueHead", target: Literal["default", "reward"]) -> None:
    r"""
    Replaces the default/reward modules in the model. The model is already unwrapped.
    """
    v_head_layer = model.v_head.summary
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [v_head_layer.weight, v_head_layer.bias]
        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    model.pretrained_model.set_adapter(target)  # set the LoRA adapter to be active
    with context_maybe_zero3:
        if target == "reward":  # save default head temporarily
            setattr(model, "default_head_weight", v_head_layer.weight.data.detach().clone())
            setattr(model, "default_head_bias", v_head_layer.bias.data.detach().clone())

        device = v_head_layer.weight.device
        v_head_layer.weight.data = model.get_buffer("{}_head_weight".format(target)).detach().clone().to(device)
        v_head_layer.bias.data = model.get_buffer("{}_head_bias".format(target)).detach().clone().to(device)


def dump_layernorm(model: "PreTrainedModel") -> Dict[str, "torch.Tensor"]:
    r"""
    Dumps the layernorm parameters in the model. The model is already unwrapped (and gathered).
    """
    layer_norm_params = {}
    for name, param in model.named_parameters():
        if param.data.dtype == torch.float32:
            layer_norm_params[name] = param.data.detach().clone()
            param.data = param.data.to(model.config.torch_dtype)

    return layer_norm_params


def restore_layernorm(model: "PreTrainedModel", layernorm_params: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
    r"""
    Restores the layernorm parameters in the model. The model is already unwrapped (and gathered).
    """
    for name, param in model.named_parameters():
        if name in layernorm_params:
            param.data = layernorm_params[name]
