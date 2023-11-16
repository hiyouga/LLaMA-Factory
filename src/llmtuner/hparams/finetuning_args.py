import json
from typing import Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class FreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for partial-parameter (freeze) fine-tuning."}
    )
    name_module_trainable: Optional[str] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for partial-parameter (freeze) fine-tuning. \
                  Use commas to separate multiple modules. \
                  LLaMA choices: [\"mlp\", \"self_attn\"], \
                  BLOOM & Falcon & ChatGLM choices: [\"mlp\", \"self_attention\"], \
                  Qwen choices: [\"mlp\", \"attn\"], \
                  Phi-1.5 choices: [\"mlp\", \"mixer\"], \
                  Others choices: the same as LLaMA."}
    )


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2.0)."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )
    lora_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  BLOOM & Falcon & ChatGLM choices: [\"query_key_value\", \"dense\", \"dense_h_to_4h\", \"dense_4h_to_h\"], \
                  Baichuan choices: [\"W_pack\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  Qwen choices: [\"c_attn\", \"attn.c_proj\", \"w1\", \"w2\", \"mlp.c_proj\"], \
                  Phi-1.5 choices: [\"Wqkv\", \"out_proj\", \"fc1\", \"fc2\"], \
                  Others choices: the same as LLaMA."}
    )
    additional_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."}
    )
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to resume training from the last LoRA weights or create new weights after merging them."}
    )


@dataclass
class RLHFArguments:
    r"""
    Arguments pertaining to the PPO and DPO training.
    """
    dpo_beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta parameter for the DPO loss."}
    )
    ppo_logger: Optional[str] = field(
        default=None,
        metadata={"help": "Log with either 'wandb' or 'tensorboard' in PPO training."}
    )
    ppo_score_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."}
    )
    ppo_target: Optional[float] = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."}
    )
    ppo_whiten_rewards: Optional[bool] = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."}
    )
    ref_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."}
    )
    ref_model_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory(s) containing the model checkpoints of the reference model."}
    )
    ref_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."}
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the reward model."}
    )
    reward_model_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory(s) containing the model checkpoints of the reward model."}
    )
    reward_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."}
    )
    reward_model_type: Optional[Literal["lora", "full"]] = field(
        default="lora",
        metadata={"help": "The checkpoint type of the reward model. The lora type only supports lora training."}
    )


@dataclass
class FinetuningArguments(FreezeArguments, LoraArguments, RLHFArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    stage: Optional[Literal["pt", "sft", "rm", "ppo", "dpo"]] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."}
    )
    finetuning_type: Optional[Literal["lora", "freeze", "full"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."}
    )
    upcast_layernorm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to upcast the layernorm weights in fp32."}
    )
    neft_alpha: Optional[float] = field(
        default=0,
        metadata={"help": "The alpha parameter to control the noise magnitude in NEFTune."}
    )
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."}
    )
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to plot the training loss after fine-tuning or not."}
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.name_module_trainable = split_arg(self.name_module_trainable)
        self.lora_alpha = self.lora_alpha or float(self.lora_rank * 2.0)
        self.lora_target = split_arg(self.lora_target)
        self.additional_target = split_arg(self.additional_target)
        self.ref_model_checkpoint = split_arg(self.ref_model_checkpoint)
        self.reward_model_checkpoint = split_arg(self.reward_model_checkpoint)

        assert self.finetuning_type in ["lora", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("Reward model is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("Lora reward model only supports lora training.")

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
