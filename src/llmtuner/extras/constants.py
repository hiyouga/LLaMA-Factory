IGNORE_INDEX = -100

VALUE_HEAD_FILE_NAME = "value_head.bin"

FINETUNING_ARGS_NAME = "finetuning_args.json"

LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp"] # for LLaMA, BLOOM and Falcon settings

METHODS = ["full", "freeze", "lora"]

SUPPORTED_MODELS = {
    "LLaMA-7B": "huggyllama/llama-7b",
    "LLaMA-13B": "huggyllama/llama-13b",
    "LLaMA-30B": "huggyllama/llama-30b",
    "LLaMA-65B": "huggyllama/llama-65b",
    "BLOOM-560M": "bigscience/bloom-560m",
    "BLOOM-3B": "bigscience/bloom-3b",
    "BLOOM-7B1": "bigscience/bloom-7b1",
    "BLOOMZ-560M": "bigscience/bloomz-560m",
    "BLOOMZ-3B": "bigscience/bloomz-3b",
    "BLOOMZ-7B1-mt": "bigscience/bloomz-7b1-mt",
    "Falcon-7B-Base": "tiiuae/falcon-7b",
    "Falcon-7B-Chat": "tiiuae/falcon-7b-instruct",
    "Falcon-40B-Base": "tiiuae/falcon-40b",
    "Falcon-40B-Chat": "tiiuae/falcon-40b-instruct",
    "Baichuan-7B": "baichuan-inc/Baichuan-7B",
    "Baichuan-13B-Base": "baichuan-inc/Baichuan-13B-Base",
    "Baichuan-13B-Chat": "baichuan-inc/Baichuan-13B-Chat",
    "InternLM-7B-Base": "internlm/internlm-7b",
    "InternLM-7B-Chat": "internlm/internlm-chat-7b"
}

DEFAULT_MODULE = { # will be deprecated
    "LLaMA": "q_proj,v_proj",
    "BLOOM": "query_key_value",
    "BLOOMZ": "query_key_value",
    "Falcon": "query_key_value",
    "Baichuan": "W_pack",
    "InternLM": "q_proj,v_proj"
}
