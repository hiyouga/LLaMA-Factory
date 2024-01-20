# coding=utf-8
# Calculates the flops of pre-trained models.
# Usage: python cal_flops.py --model_name_or_path path_to_model --batch_size 1 --seq_length 512
# Inspired by: https://www.deepspeed.ai/tutorials/flops-profiler/

from typing import Optional

import fire
import torch
from deepspeed.accelerator import get_accelerator  # type: ignore
from deepspeed.profiling.flops_profiler import get_model_profile  # type: ignore

from llmtuner import ChatModel


def calculate_flops(
    model_name_or_path: str,
    batch_size: Optional[int] = 1,
    seq_length: Optional[int] = 256,
    flash_attn: Optional[bool] = False,
):
    with get_accelerator().device(0):
        chat_model = ChatModel(dict(model_name_or_path=model_name_or_path, template="vanilla", flash_attn=flash_attn))
        fake_input = torch.ones((batch_size, seq_length), dtype=torch.long, device=chat_model.model.device)
        input_dict = {"input_ids": fake_input, "labels": fake_input.clone()}
        flops, macs, params = get_model_profile(chat_model.model, kwargs=input_dict, print_profile=True, detailed=True)
        print("FLOPs:", flops)
        print("MACs:", macs)
        print("Params:", params)


if __name__ == "__main__":
    fire.Fire(calculate_flops)
