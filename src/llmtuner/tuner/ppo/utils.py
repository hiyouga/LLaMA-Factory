import torch
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

from llmtuner.extras.constants import LAYERNORM_NAMES

if TYPE_CHECKING:
    from trl import AutoModelForCausalLMWithValueHead


def replace_model(model: "AutoModelForCausalLMWithValueHead", target: Literal["default", "reward"]) -> None:
    if target == "reward": # save default head temporarily
        valuehead_state_dict = model.v_head.state_dict()
        setattr(model, "default_head_weight", valuehead_state_dict["summary.weight"])
        setattr(model, "default_head_bias", valuehead_state_dict["summary.bias"])

    model.pretrained_model.set_adapter(target) # set the LoRA adapter to be active
    model.v_head.load_state_dict({
        "summary.weight": getattr(model, "{}_head_weight".format(target)),
        "summary.bias": getattr(model, "{}_head_bias".format(target))
    })


def cast_layernorm_dtype(
    model: "AutoModelForCausalLMWithValueHead",
    compute_dtype: torch.dtype,
    layer_norm_params: Optional[Dict[str, torch.Tensor]] = None,
    layer_norm_names: Optional[List[str]] = LAYERNORM_NAMES
) -> Tuple["AutoModelForCausalLMWithValueHead", Dict[str, torch.Tensor]]:

    layer_norm_state_dict = {}

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            if layer_norm_params is None:
                layer_norm_state_dict[name] = param.data.detach().clone() # store float32 weights for stability
                param.data = param.data.to(compute_dtype)
            else:
                param.data = layer_norm_params[name] # restore float32 weights

    return model, layer_norm_state_dict
