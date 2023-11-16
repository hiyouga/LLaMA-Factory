from collections import defaultdict, OrderedDict
from typing import Dict, Optional


CHOICES = ["A", "B", "C", "D"]

DEFAULT_MODULE = defaultdict(str)

DEFAULT_TEMPLATE = defaultdict(str)

IGNORE_INDEX = -100

LAYERNORM_NAMES = {"norm", "ln"}

LOG_FILE_NAME = "trainer_log.jsonl"

METHODS = ["full", "freeze", "lora"]

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

SUPPORTED_MODELS = OrderedDict()

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "Pre-Training": "pt"
}


def register_model_group(
    models: Dict[str, str],
    module: Optional[str] = None,
    template: Optional[str] = None
) -> None:
    prefix = None
    for name, path in models.items():
        if prefix is None:
            prefix = name.split("-")[0]
        else:
            assert prefix == name.split("-")[0], "prefix should be identical."
        SUPPORTED_MODELS[name] = path
    if module is not None:
        DEFAULT_MODULE[prefix] = module
    if template is not None:
        DEFAULT_TEMPLATE[prefix] = template


register_model_group(
    models={
        "Baichuan-7B-Base": "baichuan-inc/Baichuan-7B",
        "Baichuan-13B-Base": "baichuan-inc/Baichuan-13B-Base",
        "Baichuan-13B-Chat": "baichuan-inc/Baichuan-13B-Chat"
    },
    module="W_pack",
    template="baichuan"
)


register_model_group(
    models={
        "Baichuan2-7B-Base": "baichuan-inc/Baichuan2-7B-Base",
        "Baichuan2-13B-Base": "baichuan-inc/Baichuan2-13B-Base",
        "Baichuan2-7B-Chat": "baichuan-inc/Baichuan2-7B-Chat",
        "Baichuan2-13B-Chat": "baichuan-inc/Baichuan2-13B-Chat"
    },
    module="W_pack",
    template="baichuan2"
)


register_model_group(
    models={
        "BLOOM-560M": "bigscience/bloom-560m",
        "BLOOM-3B": "bigscience/bloom-3b",
        "BLOOM-7B1": "bigscience/bloom-7b1"
    },
    module="query_key_value"
)


register_model_group(
    models={
        "BLOOMZ-560M": "bigscience/bloomz-560m",
        "BLOOMZ-3B": "bigscience/bloomz-3b",
        "BLOOMZ-7B1-mt": "bigscience/bloomz-7b1-mt"
    },
    module="query_key_value"
)


register_model_group(
    models={
        "BlueLM-7B-Base": "vivo-ai/BlueLM-7B-Base",
        "BlueLM-7B-Chat": "vivo-ai/BlueLM-7B-Chat"
    },
    template="bluelm"
)


register_model_group(
    models={
        "ChatGLM2-6B-Chat": "THUDM/chatglm2-6b"
    },
    module="query_key_value",
    template="chatglm2"
)


register_model_group(
    models={
        "ChatGLM3-6B-Base": "THUDM/chatglm3-6b-base",
        "ChatGLM3-6B-Chat": "THUDM/chatglm3-6b"
    },
    module="query_key_value",
    template="chatglm3"
)


register_model_group(
    models={
        "ChineseLLaMA2-1.3B": "hfl/chinese-llama-2-1.3b",
        "ChineseLLaMA2-7B": "hfl/chinese-llama-2-7b",
        "ChineseLLaMA2-13B": "hfl/chinese-llama-2-13b",
        "ChineseLLaMA2-1.3B-Chat": "hfl/chinese-alpaca-2-1.3b",
        "ChineseLLaMA2-7B-Chat": "hfl/chinese-alpaca-2-7b",
        "ChineseLLaMA2-13B-Chat": "hfl/chinese-alpaca-2-13b"
    },
    template="llama2_zh"
)


register_model_group(
    models={
        "Falcon-7B": "tiiuae/falcon-7b",
        "Falcon-40B": "tiiuae/falcon-40b",
        "Falcon-180B": "tiiuae/falcon-180B",
        "Falcon-7B-Chat": "tiiuae/falcon-7b-instruct",
        "Falcon-40B-Chat": "tiiuae/falcon-40b-instruct",
        "Falcon-180B-Chat": "tiiuae/falcon-180B-chat"
    },
    module="query_key_value",
    template="falcon"
)


register_model_group(
    models={
        "InternLM-7B": "internlm/internlm-7b",
        "InternLM-20B": "internlm/internlm-20b",
        "InternLM-7B-Chat": "internlm/internlm-chat-7b",
        "InternLM-20B-Chat": "internlm/internlm-chat-20b"
    },
    template="intern"
)


register_model_group(
    models={
        "LingoWhale-8B": "deeplang-ai/LingoWhale-8B"
    },
    module="qkv_proj"
)


register_model_group(
    models={
        "LLaMA-7B": "huggyllama/llama-7b",
        "LLaMA-13B": "huggyllama/llama-13b",
        "LLaMA-30B": "huggyllama/llama-30b",
        "LLaMA-65B": "huggyllama/llama-65b"
    }
)


register_model_group(
    models={
        "LLaMA2-7B": "meta-llama/Llama-2-7b-hf",
        "LLaMA2-13B": "meta-llama/Llama-2-13b-hf",
        "LLaMA2-70B": "meta-llama/Llama-2-70b-hf",
        "LLaMA2-7B-Chat": "meta-llama/Llama-2-7b-chat-hf",
        "LLaMA2-13B-Chat": "meta-llama/Llama-2-13b-chat-hf",
        "LLaMA2-70B-Chat": "meta-llama/Llama-2-70b-chat-hf"
    },
    template="llama2"
)


register_model_group(
    models={
        "Mistral-7B": "mistralai/Mistral-7B-v0.1",
        "Mistral-7B-Chat": "mistralai/Mistral-7B-Instruct-v0.1"
    },
    template="mistral"
)


register_model_group(
    models={
        "OpenChat3.5-7B-Chat": "openchat/openchat_3.5"
    },
    template="openchat"
)


register_model_group(
    models={
        "Phi1.5-1.3B": "microsoft/phi-1_5"
    },
    module="Wqkv"
)


register_model_group(
    models={
        "Qwen-7B": "Qwen/Qwen-7B",
        "Qwen-14B": "Qwen/Qwen-14B",
        "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
        "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat"
    },
    module="c_attn",
    template="qwen"
)


register_model_group(
    models={
        "Skywork-13B-Base": "Skywork/Skywork-13B-base"
    }
)


register_model_group(
    models={
        "Vicuna1.5-7B-Chat": "lmsys/vicuna-7b-v1.5",
        "Vicuna1.5-13B-Chat": "lmsys/vicuna-13b-v1.5"
    },
    template="vicuna"
)


register_model_group(
    models={
        "XVERSE-7B": "xverse/XVERSE-7B",
        "XVERSE-13B": "xverse/XVERSE-13B",
        "XVERSE-65B": "xverse/XVERSE-65B",
        "XVERSE-7B-Chat": "xverse/XVERSE-7B-Chat",
        "XVERSE-13B-Chat": "xverse/XVERSE-13B-Chat"
    },
    template="xverse"
)


register_model_group(
    models={
        "Yayi-7B": "wenge-research/yayi-7b-llama2",
        "Yayi-13B": "wenge-research/yayi-13b-llama2"
    },
    template="yayi"
)


register_model_group(
    models={
        "Yi-6B": "01-ai/Yi-6B",
        "Yi-34B": "01-ai/Yi-34B"
    }
)


register_model_group(
    models={
        "Zephyr-7B-Alpha-Chat": "HuggingFaceH4/zephyr-7b-alpha",
        "Zephyr-7B-Beta-Chat": "HuggingFaceH4/zephyr-7b-beta"
    },
    template="zephyr"
)
