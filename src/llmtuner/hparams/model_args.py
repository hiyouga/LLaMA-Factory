from typing import Any, Dict, Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."}
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapter weight or identifier from huggingface.co/models."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."}
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
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."}
    )

    def __post_init__(self):
        self.compute_dtype = None
        self.model_max_length = None

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.adapter_name_or_path is not None: # support merging multiple lora weights
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]

        assert self.quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
