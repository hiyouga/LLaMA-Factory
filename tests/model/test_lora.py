import os

import torch

from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-LlamaForCausalLM")

TRAINING_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "llamafactory/tiny_dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_cache": True,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


def test_lora_all_modules():
    model_args, _, _, finetuning_args, _ = get_train_args(
        {
            "lora_target": "all",
            **TRAINING_ARGS,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)
    linear_modules = set()
    for name, param in model.named_parameters():
        if any(module in name for module in ["lora_A", "lora_B"]):
            linear_modules.add(name.split(".lora_", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16

    assert linear_modules == {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"}


def test_lora_extra_modules():
    model_args, _, _, finetuning_args, _ = get_train_args(
        {
            "lora_target": "all",
            "additional_target": "embed_tokens,lm_head",
            **TRAINING_ARGS,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)
    extra_modules = set()
    for name, param in model.named_parameters():
        if any(module in name for module in ["lora_A", "lora_B"]):
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        elif "modules_to_save" in name:
            extra_modules.add(name.split(".modules_to_save", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16

    assert extra_modules == {"embed_tokens", "lm_head"}
