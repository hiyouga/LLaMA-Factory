import json
from typing import Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    finetuning_type: Optional[Literal["none", "freeze", "lora", "full"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."}
    )
    num_hidden_layers: Optional[int] = field(
        default=32,
        metadata={"help": "Number of decoder blocks in the model. \
                  LLaMA choices: [\"32\", \"40\", \"60\", \"80\"], \
                  LLaMA-2 choices: [\"32\", \"40\", \"80\"], \
                  BLOOM choices: [\"24\", \"30\", \"70\"], \
                  Falcon choices: [\"32\", \"60\"], \
                  Baichuan choices: [\"32\", \"40\"] \
                  Qwen choices: [\"32\"]"}
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for Freeze fine-tuning."}
    )
    name_module_trainable: Optional[Literal["mlp", "self_attn", "self_attention"]] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for Freeze fine-tuning. \
                  LLaMA & LLaMA-2 choices: [\"mlp\", \"self_attn\"], \
                  BLOOM & Falcon choices: [\"mlp\", \"self_attention\"], \
                  Baichuan choices: [\"mlp\", \"self_attn\"], \
                  Qwen choices: [\"attn\", \"mlp\"]"}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA fine-tuning (similar with the learning rate)."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )
    lora_target: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules. \
                  LLaMA & LLaMA-2 & InternLM choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  BLOOM & Falcon choices: [\"query_key_value\", \"self_attention.dense\", \"mlp.dense\"], \
                  Baichuan choices: [\"W_pack\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  Qwen choices: [\"c_attn\", \"c_proj\", \"w1\", \"w2\"]"}
    )

    def __post_init__(self):
        if isinstance(self.lora_target, str): # support custom target modules/layers of LoRA
            self.lora_target = [target.strip() for target in self.lora_target.split(",")]

        if self.num_layer_trainable > 0: # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [self.num_hidden_layers - k - 1 for k in range(self.num_layer_trainable)]
        else: # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-self.num_layer_trainable)]

        self.trainable_layers = ["{:d}.{}".format(idx, self.name_module_trainable) for idx in trainable_layer_ids]

        assert self.finetuning_type in ["none", "freeze", "lora", "full"], "Invalid fine-tuning method."

    def save_to_json(self, json_path: str):
        """Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
