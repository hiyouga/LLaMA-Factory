import os
from collections import defaultdict, OrderedDict
from typing import Dict, Optional, Union

CHOICES = ["A", "B", "C", "D"]

DEFAULT_MODULE = defaultdict(str)

DEFAULT_TEMPLATE = defaultdict(str)

IGNORE_INDEX = -100

LAYERNORM_NAMES = {"norm", "ln"}

LOG_FILE_NAME = "trainer_log.jsonl"

METHODS = ["full", "freeze", "lora"]

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

SUPPORTED_MODELS = OrderedDict()

MODELSCOPE_MODELS = OrderedDict()

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "Pre-Training": "pt"
}


def register_model_group(
    models: Dict[str, Union[str, Dict[str, str]]],
    module: Optional[str] = None,
    template: Optional[str] = None
) -> None:
    prefix = None
    for name, path in models.items():
        if prefix is None:
            prefix = name.split("-")[0]
        else:
            assert prefix == name.split("-")[0], "prefix should be identical."

        if not os.environ.get('USE_MODELSCOPE_HUB', False):
            # If path is a string, we treat it as a huggingface model-id by default.
            SUPPORTED_MODELS[name] = path["hf"] if isinstance(path, dict) else path
        elif isinstance(path, dict) and "ms" in path:
            # Use ModelScope modelhub
            SUPPORTED_MODELS[name] = path["ms"]
    if module is not None:
        DEFAULT_MODULE[prefix] = module
    if template is not None:
        DEFAULT_TEMPLATE[prefix] = template


register_model_group(
    models={
        "Baichuan-7B-Base": {
            "hf": "baichuan-inc/Baichuan-7B",
            "ms": "baichuan-inc/baichuan-7B",
        },
        "Baichuan-13B-Base": {
            "hf": "baichuan-inc/Baichuan-13B-Base",
            "ms": "baichuan-inc/Baichuan-13B-Base",
        },
        "Baichuan-13B-Chat": {
            "hf": "baichuan-inc/Baichuan-13B-Chat",
            "ms": "baichuan-inc/Baichuan-13B-Base",
        }
    },
    module="W_pack",
    template="baichuan"
)


register_model_group(
    models={
        "Baichuan2-7B-Base": {
            "hf": "baichuan-inc/Baichuan2-7B-Base",
            "ms": "baichuan-inc/Baichuan2-7B-Base",
        },
        "Baichuan2-13B-Base": {
            "hf": "baichuan-inc/Baichuan2-13B-Base",
            "ms": "baichuan-inc/Baichuan2-13B-Base",
        },
        "Baichuan2-7B-Chat": {
            "hf": "baichuan-inc/Baichuan2-7B-Chat",
            "ms": "baichuan-inc/Baichuan2-7B-Chat",
        },
        "Baichuan2-13B-Chat": {
            "hf": "baichuan-inc/Baichuan2-13B-Chat",
            "ms": "baichuan-inc/Baichuan2-13B-Chat",
        }
    },
    module="W_pack",
    template="baichuan2"
)


register_model_group(
    models={
        "BLOOM-560M": {
            "hf": "bigscience/bloom-560m",
            "ms": "AI-ModelScope/bloom-560m",
        },
        "BLOOM-3B": {
            "hf": "bigscience/bloom-3b",
            "ms": "AI-ModelScope/bloom-3b",
        },
        "BLOOM-7B1": {
            "hf": "bigscience/bloom-7b1",
            "ms": "AI-ModelScope/bloom-7b1",
        }
    },
    module="query_key_value"
)


register_model_group(
    models={
        "BLOOMZ-560M": {
            "hf": "bigscience/bloomz-560m",
            "ms": "AI-ModelScope/bloomz-560m",
        },
        "BLOOMZ-3B": {
            "hf": "bigscience/bloomz-3b",
            "ms": "AI-ModelScope/bloomz-3b",
        },
        "BLOOMZ-7B1-mt": {
            "hf": "bigscience/bloomz-7b1-mt",
            "ms": "AI-ModelScope/bloomz-7b1-mt",
        }
    },
    module="query_key_value"
)


register_model_group(
    models={
        "BlueLM-7B-Base": {
            "hf": "vivo-ai/BlueLM-7B-Base",
            "ms": "vivo-ai/BlueLM-7B-Base",
        },
        "BlueLM-7B-Chat": {
            "hf": "vivo-ai/BlueLM-7B-Chat",
            "ms": "vivo-ai/BlueLM-7B-Chat",
        }
    },
    template="bluelm"
)


register_model_group(
    models={
        "ChatGLM2-6B-Chat": {
            "hf": "THUDM/chatglm2-6b",
            "ms": "ZhipuAI/chatglm2-6b",
        }
    },
    module="query_key_value",
    template="chatglm2"
)


register_model_group(
    models={
        "ChatGLM3-6B-Base": {
            "hf": "THUDM/chatglm3-6b-base",
            "ms": "ZhipuAI/chatglm3-6b-base",
        },
        "ChatGLM3-6B-Chat": {
            "hf": "THUDM/chatglm3-6b",
            "ms": "ZhipuAI/chatglm3-6b",
        }
    },
    module="query_key_value",
    template="chatglm3"
)


register_model_group(
    models={
        "ChineseLLaMA2-1.3B": {
            "hf": "hfl/chinese-llama-2-1.3b",
            "ms": "AI-ModelScope/chinese-llama-2-1.3b",
        },
        "ChineseLLaMA2-7B": {
            "hf": "hfl/chinese-llama-2-7b",
            "ms": "AI-ModelScope/chinese-llama-2-7b",
        },
        "ChineseLLaMA2-13B": {
            "hf": "hfl/chinese-llama-2-13b",
            "ms": "AI-ModelScope/chinese-llama-2-13b",
        },
        "ChineseLLaMA2-1.3B-Chat": {
            "hf": "hfl/chinese-alpaca-2-1.3b",
            "ms": "AI-ModelScope/chinese-alpaca-2-1.3b",
        },
        "ChineseLLaMA2-7B-Chat": {
            "hf": "hfl/chinese-alpaca-2-7b",
            "ms": "AI-ModelScope/chinese-alpaca-2-7b",
        },
        "ChineseLLaMA2-13B-Chat": {
            "hf": "hfl/chinese-alpaca-2-13b",
            "ms": "AI-ModelScope/chinese-alpaca-2-13b",
        }
    },
    template="llama2_zh"
)


register_model_group(
    models={
        "Falcon-7B": {
            "hf": "tiiuae/falcon-7b",
            "ms": "AI-ModelScope/falcon-7b",
        },
        "Falcon-40B": {
            "hf": "tiiuae/falcon-40b",
            "ms": "AI-ModelScope/falcon-40b",
        },
        "Falcon-180B": {
            "hf": "tiiuae/falcon-180B",
            "ms": "AI-ModelScope/falcon-180B",
        },
        "Falcon-7B-Chat": {
            "hf": "tiiuae/falcon-7b-instruct",
            "ms": "AI-ModelScope/falcon-7b-instruct",
        },
        "Falcon-40B-Chat": {
            "hf": "tiiuae/falcon-40b-instruct",
            "ms": "AI-ModelScope/falcon-40b-instruct",
        },
        "Falcon-180B-Chat": {
            "hf": "tiiuae/falcon-180B-chat",
            "ms": "AI-ModelScope/falcon-180B-chat",
        }
    },
    module="query_key_value",
    template="falcon"
)


register_model_group(
    models={
        "InternLM-7B": {
            "hf": "internlm/internlm-7b",
            "ms": "Shanghai_AI_Laboratory/internlm-7b",
        },
        "InternLM-20B": {
            "hf": "internlm/internlm-20b",
            "ms": "Shanghai_AI_Laboratory/internlm-20b",
        },
        "InternLM-7B-Chat": {
            "hf": "internlm/internlm-chat-7b",
            "ms": "Shanghai_AI_Laboratory/internlm-chat-7b",
        },
        "InternLM-20B-Chat": {
            "hf": "internlm/internlm-chat-20b",
            "ms": "Shanghai_AI_Laboratory/internlm-chat-20b",
        }
    },
    template="intern"
)


register_model_group(
    models={
        "LingoWhale-8B": {
            "hf": "deeplang-ai/LingoWhale-8B",
            "ms": "DeepLang/LingoWhale-8B",
        }
    },
    module="qkv_proj"
)


register_model_group(
    models={
        "LLaMA-7B": {
            "hf": "huggyllama/llama-7b",
            "ms": "skyline2006/llama-7b",
        },
        "LLaMA-13B": {
            "hf": "huggyllama/llama-13b",
            "ms": "skyline2006/llama-13b",
        },
        "LLaMA-30B": {
            "hf": "huggyllama/llama-30b",
            "ms": "skyline2006/llama-30b",
        },
        "LLaMA-65B": {
            "hf": "huggyllama/llama-65b",
            "ms": "skyline2006/llama-65b",
        }
    }
)


register_model_group(
    models={
        "LLaMA2-7B": {
            "hf": "meta-llama/Llama-2-7b-hf",
            "ms": "modelscope/Llama-2-7b-ms",
        },
        "LLaMA2-13B": {
            "hf": "meta-llama/Llama-2-13b-hf",
            "ms": "modelscope/Llama-2-13b-ms",
        },
        "LLaMA2-70B": {
            "hf": "meta-llama/Llama-2-70b-hf",
            "ms": "modelscope/Llama-2-70b-ms",
        },
        "LLaMA2-7B-Chat": {
            "hf": "meta-llama/Llama-2-7b-chat-hf",
            "ms": "modelscope/Llama-2-7b-chat-ms",
        },
        "LLaMA2-13B-Chat": {
            "hf": "meta-llama/Llama-2-13b-chat-hf",
            "ms": "modelscope/Llama-2-13b-chat-ms",
        },
        "LLaMA2-70B-Chat": {
            "hf": "meta-llama/Llama-2-70b-chat-hf",
            "ms": "modelscope/Llama-2-70b-chat-ms",
        }
    },
    template="llama2"
)


register_model_group(
    models={
        "Mistral-7B": {
            "hf": "mistralai/Mistral-7B-v0.1",
            "ms": "AI-ModelScope/Mistral-7B-v0.1",
        },
        "Mistral-7B-Chat": {
            "hf": "mistralai/Mistral-7B-Instruct-v0.1",
            "ms": "AI-ModelScope/Mistral-7B-Instruct-v0.1",
        }
    },
    template="mistral"
)


register_model_group(
    models={
        "OpenChat3.5-7B-Chat": {
            "hf": "openchat/openchat_3.5",
            "ms": "myxiongmodel/openchat_3.5",
        }
    },
    template="openchat"
)


register_model_group(
    models={
        "Phi1.5-1.3B": {
            "hf": "microsoft/phi-1_5",
            "ms": "allspace/PHI_1-5",
        }
    },
    module="Wqkv"
)


register_model_group(
    models={
        "Qwen-7B": {
            "hf": "Qwen/Qwen-7B",
            "ms": "qwen/Qwen-7B",
        },
        "Qwen-14B": {
            "hf": "Qwen/Qwen-14B",
            "ms": "qwen/Qwen-14B",
        },
        "Qwen-7B-Chat": {
            "hf": "Qwen/Qwen-7B-Chat",
            "ms": "qwen/Qwen-7B-Chat",
        },
        "Qwen-14B-Chat": {
            "hf": "Qwen/Qwen-14B-Chat",
            "ms": "qwen/Qwen-14B-Chat",
        },
        "Qwen-7B-int8-Chat": {
            "hf": "Qwen/Qwen-7B-Chat-Int8",
            "ms": "qwen/Qwen-7B-Chat-Int8",
        },
        "Qwen-7B-int4-Chat": {
            "hf": "Qwen/Qwen-7B-Chat-Int4",
            "ms": "qwen/Qwen-7B-Chat-Int4",
        },
        "Qwen-14B-int8-Chat": {
            "hf": "Qwen/Qwen-14B-Chat-Int8",
            "ms": "qwen/Qwen-14B-Chat-Int8",
        },
        "Qwen-14B-int4-Chat": {
            "hf": "Qwen/Qwen-14B-Chat-Int4",
            "ms": "qwen/Qwen-14B-Chat-Int4",
        }
    },
    module="c_attn",
    template="qwen"
)


register_model_group(
    models={
        "Skywork-13B-Base": {
            "hf": "Skywork/Skywork-13B-base",
            "ms": "skywork/Skywork-13B-base",
        }
    }
)


register_model_group(
    models={
        "Vicuna1.5-7B-Chat": {
            "hf": "lmsys/vicuna-7b-v1.5",
            "ms": "AI-ModelScope/vicuna-7b-v1.5",
        },
        "Vicuna1.5-13B-Chat": {
            "hf": "lmsys/vicuna-13b-v1.5",
            "ms": "Xorbits/vicuna-13b-v1.5",
        }
    },
    template="vicuna"
)


register_model_group(
    models={
        "XVERSE-7B": {
            "hf": "xverse/XVERSE-7B",
            "ms": "xverse/XVERSE-7B",
        },
        "XVERSE-13B": {
            "hf": "xverse/XVERSE-13B",
            "ms": "xverse/XVERSE-13B",
        },
        "XVERSE-65B": {
            "hf": "xverse/XVERSE-65B",
            "ms": "xverse/XVERSE-65B",
        },
        "XVERSE-7B-Chat": {
            "hf": "xverse/XVERSE-7B-Chat",
            "ms": "xverse/XVERSE-7B-Chat",
        },
        "XVERSE-13B-Chat": {
            "hf": "xverse/XVERSE-13B-Chat",
            "ms": "xverse/XVERSE-13B-Chat",
        }
    },
    template="xverse"
)


register_model_group(
    models={
        "Yayi-7B": {
            "hf": "wenge-research/yayi-7b-llama2",
            "ms": "AI-ModelScope/yayi-7b-llama2",
        },
        "Yayi-13B": {
            "hf": "wenge-research/yayi-13b-llama2",
            "ms": "AI-ModelScope/yayi-13b-llama2",
        }
    },
    template="yayi"
)


register_model_group(
    models={
        "Yi-6B": {
            "hf": "01-ai/Yi-6B",
            "ms": "01ai/Yi-6B",
        },
        "Yi-34B": {
            "hf": "01-ai/Yi-34B",
            "ms": "01ai/Yi-34B",
        },
        "Yi-34B-Chat": {
            "hf": "01-ai/Yi-34B-Chat",
            "ms": "01ai/Yi-34B-Chat",
        },
        "Yi-34B-int8-Chat": {
            "hf": "01-ai/Yi-34B-Chat-8bits",
            "ms": "01ai/Yi-34B-Chat-8bits",
        }
    },
    template="yi"
)


register_model_group(
    models={
        "Zephyr-7B-Alpha-Chat": {
            "hf": "HuggingFaceH4/zephyr-7b-alpha",
            "ms": "AI-ModelScope/zephyr-7b-alpha",
        },
        "Zephyr-7B-Beta-Chat": {
            "hf": "HuggingFaceH4/zephyr-7b-beta",
            "ms": "modelscope/zephyr-7b-beta",
        }
    },
    template="zephyr"
)
