from typing import TYPE_CHECKING, Literal

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
