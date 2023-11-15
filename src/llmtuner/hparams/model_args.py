from typing import Any, Dict, Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co."}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    split_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."}
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )
    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."}
    )
    double_quantization: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use double quantization in int4 training or not."}
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Adopt scaled rotary positional embeddings."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory(s) containing the model checkpoints as well as the configurations."}
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    shift_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."}
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."}
    )

    def __post_init__(self):
        self.compute_dtype = None
        self.model_max_length = None

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.checkpoint_dir is not None: # support merging multiple lora weights
            self.checkpoint_dir = [cd.strip() for cd in self.checkpoint_dir.split(",")]

        assert self.quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
