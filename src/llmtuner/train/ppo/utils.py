import json
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch

from ...extras.packages import is_requests_available


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead

if is_requests_available():
    import requests


def get_rewards_from_server(server_url: str, messages: List[str]) -> List[torch.Tensor]:
    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(server_url, json=payload, headers=headers)
    rewards = json.loads(response.text)["scores"]
    return torch.Tensor(rewards)


def replace_model(model: "AutoModelForCausalLMWithValueHead", target: Literal["default", "reward"]) -> None:
    if target == "reward":  # save default head temporarily
        valuehead_state_dict: Dict[str, torch.Tensor] = model.v_head.state_dict()
        setattr(model, "default_head_weight", valuehead_state_dict["summary.weight"].detach().clone())
        setattr(model, "default_head_bias", valuehead_state_dict["summary.bias"].detach().clone())

    model.pretrained_model.set_adapter(target)  # set the LoRA adapter to be active
    model.v_head.load_state_dict(
        {
            "summary.weight": model.get_buffer("{}_head_weight".format(target)).detach().clone(),
            "summary.bias": model.get_buffer("{}_head_bias".format(target)).detach().clone(),
        }
    )


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
