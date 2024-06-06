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


def get_rewards_from_server(server_url: str, messages: List[str]) -> List[torch.Tensor]:
    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(server_url, json=payload, headers=headers)
    rewards = json.loads(response.text)["scores"]
    return torch.Tensor(rewards)


def replace_model(model: "AutoModelForCausalLMWithValueHead", target: Literal["default", "reward"]) -> None:
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [model.v_head.summary.weight, model.v_head.summary.bias]
        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        if target == "reward":  # save default head temporarily
            setattr(model, "default_head_weight", model.v_head.summary.weight.data.detach().clone())
            setattr(model, "default_head_bias", model.v_head.summary.bias.data.detach().clone())

        model.pretrained_model.set_adapter(target)  # set the LoRA adapter to be active
        model.v_head.summary.weight.data = model.get_buffer("{}_head_weight".format(target)).detach().clone()
        model.v_head.summary.bias.data = model.get_buffer("{}_head_bias".format(target)).detach().clone()


def dump_layernorm(model: "PreTrainedModel") -> Dict[str, torch.Tensor]:
    layer_norm_params = {}
    for name, param in model.named_parameters():
        if param.data.dtype == torch.float32:
            layer_norm_params[name] = param.data.detach().clone()
            param.data = param.data.to(model.config.torch_dtype)

    return layer_norm_params


def restore_layernorm(model: "PreTrainedModel", layernorm_params: Optional[Dict[str, torch.Tensor]] = None) -> None:
    for name, param in model.named_parameters():
        if name in layernorm_params:
            param.data = layernorm_params[name]
