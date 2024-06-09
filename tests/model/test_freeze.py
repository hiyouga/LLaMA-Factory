import os

import torch

from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-LlamaForCausalLM")

TRAINING_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "freeze",
    "dataset": "llamafactory/tiny_dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_cache": True,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


def test_freeze_all_modules():
    model_args, _, _, finetuning_args, _ = get_train_args(
        {
            "freeze_trainable_layers": 1,
            **TRAINING_ARGS,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)
    for name, param in model.named_parameters():
        if name.startswith("model.layers.1."):
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16


def test_freeze_extra_modules():
    model_args, _, _, finetuning_args, _ = get_train_args(
        {
            "freeze_trainable_layers": 1,
            "freeze_extra_modules": "embed_tokens,lm_head",
            **TRAINING_ARGS,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)
    for name, param in model.named_parameters():
        if name.startswith("model.layers.1.") or any(module in name for module in ["embed_tokens", "lm_head"]):
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16
