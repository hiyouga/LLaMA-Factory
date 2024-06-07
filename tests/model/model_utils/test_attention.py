import os

from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available

from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-LlamaForCausalLM")


def test_attention():
    attention_available = ["off"]
    if is_torch_sdpa_available():
        attention_available.append("sdpa")

    if is_flash_attn_2_available():
        attention_available.append("fa2")

    llama_attention_classes = {
        "off": "LlamaAttention",
        "sdpa": "LlamaSdpaAttention",
        "fa2": "LlamaFlashAttention2",
    }
    for requested_attention in attention_available:
        model_args, _, finetuning_args, _ = get_infer_args(
            {
                "model_name_or_path": TINY_LLAMA,
                "template": "llama2",
                "flash_attn": requested_attention,
            }
        )
        tokenizer_module = load_tokenizer(model_args)
        model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args)
        for module in model.modules():
            if "Attention" in module.__class__.__name__:
                assert module.__class__.__name__ == llama_attention_classes[requested_attention]
