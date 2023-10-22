import json
from typing import Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class FinetuningArguments:
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    stage: Optional[Literal["pt", "sft", "rm", "ppo", "dpo"]] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."}
    )
    finetuning_type: Optional[Literal["lora", "freeze", "full", "none"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."}
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for partial-parameter (freeze) fine-tuning."}
    )
    name_module_trainable: Optional[Literal["mlp", "self_attn", "self_attention"]] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for partial-parameter (freeze) fine-tuning. \
                  LLaMA choices: [\"mlp\", \"self_attn\"], \
                  BLOOM & Falcon & ChatGLM2 choices: [\"mlp\", \"self_attention\"], \
                  Qwen choices: [\"mlp\", \"attn\"], \
                  Phi-1.5 choices: [\"mlp\", \"mixer\"], \
                  LLaMA-2, Baichuan, InternLM, XVERSE choices: the same as LLaMA."}
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
        default=None,
        metadata={"help": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  BLOOM & Falcon & ChatGLM2 choices: [\"query_key_value\", \"self_attention.dense\", \"mlp.dense\"], \
                  Baichuan choices: [\"W_pack\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  Qwen choices: [\"c_attn\", \"attn.c_proj\", \"w1\", \"w2\", \"mlp.c_proj\"], \
                  Phi-1.5 choices: [\"Wqkv\", \"out_proj\", \"fc1\", \"fc2\"], \
                  LLaMA-2, InternLM, XVERSE choices: the same as LLaMA."}
    )
    additional_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."}
    )
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to resume training from the last LoRA weights or create new weights after merging them."}
    )
    ppo_score_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."}
    )
    ppo_logger: Optional[str] = field(
        default=None,
        metadata={"help": "Log with either 'wandb' or 'tensorboard' in PPO training."}
    )
    ppo_target: Optional[float] = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."}
    )
    dpo_beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta parameter for the DPO loss."}
    )
    upcast_layernorm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to upcast the layernorm weights in fp32."}
    )
    neft_alpha: Optional[float] = field(
        default=0,
        metadata={"help": "The alpha parameter to control the noise magnitude in NEFTune."}
    )

    def __post_init__(self):
        if isinstance(self.lora_target, str): # support custom target modules/layers of LoRA
            self.lora_target = [target.strip() for target in self.lora_target.split(",")]

        if isinstance(self.additional_target, str):
            self.additional_target = [target.strip() for target in self.additional_target.split(",")]

        assert self.finetuning_type in ["lora", "freeze", "full", "none"], "Invalid fine-tuning method."

    def save_to_json(self, json_path: str):
        r"""Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        r"""Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
