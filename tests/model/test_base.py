import os

import torch
from transformers import AutoModelForCausalLM

from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "template": "llama3",
    "infer_dtype": "float16",
}


def compare_model(model_a: "torch.nn.Module", model_b: "torch.nn.Module"):
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    assert set(state_dict_a.keys()) == set(state_dict_b.keys())
    for name in state_dict_a.keys():
        assert torch.allclose(state_dict_a[name], state_dict_b[name]) is True


def test_base():
    model_args, _, finetuning_args, _ = get_infer_args(INFER_ARGS)
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=False)
    ref_model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA, torch_dtype=model.dtype, device_map=model.device)
    compare_model(model, ref_model)
