# copied from https://github.com/alibaba/ROLL/blob/main/mcore_adapter/tools/convert.py
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from mcore_adapter.models.converter.post_converter import convert_checkpoint_to_hf, convert_checkpoint_to_mca
from mcore_adapter.training_args import DistributingParallelArguments
from mcore_adapter.utils import get_logger
from transformers import AutoConfig, HfArgumentParser


logger = get_logger(__name__)


@dataclass
class ConvertArguments:
    checkpoint_path: str
    output_path: str = field(default="./output")
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    convert_model_max_length: Optional[int] = field(
        default=None, metadata={"help": "Change the model_max_length in hf config.json ."}
    )

    def __post_init__(self):
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot be both True.")

def convert_mca_to_hf(convert_args: ConvertArguments):
    torch_dtype = None
    if convert_args.bf16:
        torch_dtype = torch.bfloat16
    elif convert_args.fp16:
        torch_dtype = torch.float16
    convert_checkpoint_to_hf(convert_args.checkpoint_path, convert_args.output_path, torch_dtype=torch_dtype)

    if convert_args.convert_model_max_length is not None:
        config = AutoConfig.from_pretrained(convert_args.output_path, trust_remote_code=True)
        config.model_max_length = convert_args.convert_model_max_length
        config.save_pretrained(convert_args.output_path)

def main():
    convert_args, dist_args = HfArgumentParser(
        [ConvertArguments, DistributingParallelArguments]
    ).parse_args_into_dataclasses()

    mca_config_path = os.path.join(convert_args.checkpoint_path, "mca_config.json")
    from_mca = os.path.exists(mca_config_path)

    if not from_mca:
        convert_checkpoint_to_mca(
            convert_args.checkpoint_path,
            convert_args.output_path,
            dist_args,
            bf16=convert_args.bf16,
            fp16=convert_args.fp16,
        )
    else:
        convert_mca_to_hf(convert_args)


if __name__ == "__main__":
    main()
