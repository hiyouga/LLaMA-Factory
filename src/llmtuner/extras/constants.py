from enum import Enum
from collections import defaultdict, OrderedDict
from typing import Dict, Optional


CHOICES = ["A", "B", "C", "D"]

DEFAULT_MODULE = defaultdict(str)

DEFAULT_TEMPLATE = defaultdict(str)

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text"
}

IGNORE_INDEX = -100

LAYERNORM_NAMES = {"norm", "ln"}

LOG_FILE_NAME = "trainer_log.jsonl"

METHODS = ["full", "freeze", "lora"]

PEFT_METHODS = ["lora"]

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

SUPPORTED_MODELS = OrderedDict()

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "Pre-Training": "pt"
}

class DownloadSource(str, Enum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"


def register_model_group(
    models: Dict[str, Dict[DownloadSource, str]],
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
        "Baichuan-7B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-7B",
            DownloadSource.MODELSCOPE: "baichuan-inc/baichuan-7B"
        },
        "Baichuan-13B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-13B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan-13B-Base"
        },
        "Baichuan-13B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-13B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan-13B-Chat"
        }
    },
    module="W_pack",
    template="baichuan"
)


register_model_group(
    models={
        "Baichuan2-7B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-7B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-7B-Base"
        },
        "Baichuan2-13B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-13B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-13B-Base"
        },
        "Baichuan2-7B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-7B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-7B-Chat"
        },
        "Baichuan2-13B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-13B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-13B-Chat"
        }
    },
    module="W_pack",
    template="baichuan2"
)


register_model_group(
    models={
        "BLOOM-560M": {
            DownloadSource.DEFAULT: "bigscience/bloom-560m",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-560m"
        },
        "BLOOM-3B": {
            DownloadSource.DEFAULT: "bigscience/bloom-3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-3b"
        },
        "BLOOM-7B1": {
            DownloadSource.DEFAULT: "bigscience/bloom-7b1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-7b1"
        }
    },
    module="query_key_value"
)


register_model_group(
    models={
        "BLOOMZ-560M": {
            DownloadSource.DEFAULT: "bigscience/bloomz-560m",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-560m"
        },
        "BLOOMZ-3B": {
            DownloadSource.DEFAULT: "bigscience/bloomz-3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-3b"
        },
        "BLOOMZ-7B1-mt": {
            DownloadSource.DEFAULT: "bigscience/bloomz-7b1-mt",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-7b1-mt"
        }
    },
    module="query_key_value"
)


register_model_group(
    models={
        "BlueLM-7B-Base": {
            DownloadSource.DEFAULT: "vivo-ai/BlueLM-7B-Base",
            DownloadSource.MODELSCOPE: "vivo-ai/BlueLM-7B-Base"
        },
        "BlueLM-7B-Chat": {
            DownloadSource.DEFAULT: "vivo-ai/BlueLM-7B-Chat",
            DownloadSource.MODELSCOPE: "vivo-ai/BlueLM-7B-Chat"
        }
    },
    template="bluelm"
)


register_model_group(
    models={
        "ChatGLM2-6B-Chat": {
            DownloadSource.DEFAULT: "THUDM/chatglm2-6b",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm2-6b"
        }
    },
    module="query_key_value",
    template="chatglm2"
)


register_model_group(
    models={
        "ChatGLM3-6B-Base": {
            DownloadSource.DEFAULT: "THUDM/chatglm3-6b-base",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm3-6b-base"
        },
        "ChatGLM3-6B-Chat": {
            DownloadSource.DEFAULT: "THUDM/chatglm3-6b",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm3-6b"
        }
    },
    module="query_key_value",
    template="chatglm3"
)


register_model_group(
    models={
        "ChineseLLaMA2-1.3B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-1.3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-1.3b"
        },
        "ChineseLLaMA2-7B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-7b"
        },
        "ChineseLLaMA2-13B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-13b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-13b"
        },
        "ChineseLLaMA2-1.3B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-1.3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-1.3b"
        },
        "ChineseLLaMA2-7B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-7b"
        },
        "ChineseLLaMA2-13B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-13b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-13b"
        }
    },
    template="llama2_zh"
)


register_model_group(
    models={
        "DeepseekLLM-7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-7b-base"
        },
        "DeepseekLLM-67B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-67b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-67b-base"
        },
        "DeepseekLLM-7B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-7b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-7b-chat"
        },
        "DeepseekLLM-67B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-67b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-67b-chat"
        }
    },
    template="deepseek"
)


register_model_group(
    models={
        "DeepseekCoder-6.7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-6.7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-6.7b-base"
        },
        "DeepseekCoder-33B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-33b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-33b-base"
        },
        "DeepseekCoder-6.7B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-6.7b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-6.7b-instruct"
        },
        "DeepseekCoder-33B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-33b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-33b-instruct"
        }
    },
    template="deepseekcoder"
)


register_model_group(
    models={
        "Falcon-7B": {
            DownloadSource.DEFAULT: "tiiuae/falcon-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-7b"
        },
        "Falcon-40B": {
            DownloadSource.DEFAULT: "tiiuae/falcon-40b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-40b"
        },
        "Falcon-180B": {
            DownloadSource.DEFAULT: "tiiuae/falcon-180b",
            DownloadSource.MODELSCOPE: "modelscope/falcon-180B"
        },
        "Falcon-7B-Chat": {
            DownloadSource.DEFAULT: "tiiuae/falcon-7b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-7b-instruct"
        },
        "Falcon-40B-Chat": {
            DownloadSource.DEFAULT: "tiiuae/falcon-40b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-40b-instruct"
        },
        "Falcon-180B-Chat": {
            DownloadSource.DEFAULT: "tiiuae/falcon-180b-chat",
            DownloadSource.MODELSCOPE: "modelscope/falcon-180B-chat"
        }
    },
    module="query_key_value",
    template="falcon"
)


register_model_group(
    models={
        "InternLM-7B": {
            DownloadSource.DEFAULT: "internlm/internlm-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-7b"
        },
        "InternLM-20B": {
            DownloadSource.DEFAULT: "internlm/internlm-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-20b"
        },
        "InternLM-7B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm-chat-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-chat-7b"
        },
        "InternLM-20B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm-chat-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-chat-20b"
        }
    },
    template="intern"
)


register_model_group(
    models={
        "LingoWhale-8B": {
            DownloadSource.DEFAULT: "deeplang-ai/LingoWhale-8B",
            DownloadSource.MODELSCOPE: "DeepLang/LingoWhale-8B"
        }
    },
    module="qkv_proj"
)


register_model_group(
    models={
        "LLaMA-7B": {
            DownloadSource.DEFAULT: "huggyllama/llama-7b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-7b"
        },
        "LLaMA-13B": {
            DownloadSource.DEFAULT: "huggyllama/llama-13b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-13b"
        },
        "LLaMA-30B": {
            DownloadSource.DEFAULT: "huggyllama/llama-30b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-30b"
        },
        "LLaMA-65B": {
            DownloadSource.DEFAULT: "huggyllama/llama-65b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-65b"
        }
    }
)


register_model_group(
    models={
        "LLaMA2-7B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-7b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-7b-ms"
        },
        "LLaMA2-13B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-13b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-13b-ms"
        },
        "LLaMA2-70B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-70b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-70b-ms"
        },
        "LLaMA2-7B-Chat": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-7b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-7b-chat-ms"
        },
        "LLaMA2-13B-Chat": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-13b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-13b-chat-ms"
        },
        "LLaMA2-70B-Chat": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-70b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-70b-chat-ms"
        }
    },
    template="llama2"
)


register_model_group(
    models={
        "Mistral-7B": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-v0.1"
        },
        "Mistral-7B-Chat": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-Instruct-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-Instruct-v0.1"
        },
        "Mistral-7B-v0.2-Chat": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-Instruct-v0.2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-Instruct-v0.2"
        }
    },
    template="mistral"
)


register_model_group(
    models={
        "Mixtral-8x7B": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x7B-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x7B-v0.1"
        },
        "Mixtral-8x7B-Chat": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x7B-Instruct-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1"
        }
    },
    template="mistral"
)


register_model_group(
    models={
        "OpenChat3.5-7B-Chat": {
            DownloadSource.DEFAULT: "openchat/openchat_3.5",
            DownloadSource.MODELSCOPE: "myxiongmodel/openchat_3.5"
        }
    },
    template="openchat"
)


register_model_group(
    models={
        "Phi1.5-1.3B": {
            DownloadSource.DEFAULT: "microsoft/phi-1_5",
            DownloadSource.MODELSCOPE: "allspace/PHI_1-5"
        }
    },
    module="Wqkv"
)


register_model_group(
    models={
        "Qwen-1.8B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B",
            DownloadSource.MODELSCOPE: "qwen/Qwen-1_8B"
        },
        "Qwen-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B",
            DownloadSource.MODELSCOPE: "qwen/Qwen-7B"
        },
        "Qwen-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B",
            DownloadSource.MODELSCOPE: "qwen/Qwen-14B"
        },
        "Qwen-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B",
            DownloadSource.MODELSCOPE: "qwen/Qwen-72B"
        },
        "Qwen-1.8B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen-1_8B-Chat"
        },
        "Qwen-7B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen-7B-Chat"
        },
        "Qwen-14B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen-14B-Chat"
        },
        "Qwen-72B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen-72B-Chat"
        },
        "Qwen-1.8B-int8-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen-1_8B-Chat-Int8"
        },
        "Qwen-1.8B-int4-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen-1_8B-Chat-Int4"
        },
        "Qwen-7B-int8-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen-7B-Chat-Int8"
        },
        "Qwen-7B-int4-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen-7B-Chat-Int4"
        },
        "Qwen-14B-int8-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen-14B-Chat-Int8"
        },
        "Qwen-14B-int4-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen-14B-Chat-Int4"
        },
        "Qwen-72B-int8-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen-72B-Chat-Int8"
        },
        "Qwen-72B-int4-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen-72B-Chat-Int4"
        }
    },
    module="c_attn",
    template="qwen"
)


register_model_group(
    models={
        "Skywork-13B-Base": {
            DownloadSource.DEFAULT: "Skywork/Skywork-13B-base",
            DownloadSource.MODELSCOPE: "skywork/Skywork-13B-base"
        }
    }
)


register_model_group(
    models={
        "Vicuna1.5-7B-Chat": {
            DownloadSource.DEFAULT: "lmsys/vicuna-7b-v1.5",
            DownloadSource.MODELSCOPE: "Xorbits/vicuna-7b-v1.5"
        },
        "Vicuna1.5-13B-Chat": {
            DownloadSource.DEFAULT: "lmsys/vicuna-13b-v1.5",
            DownloadSource.MODELSCOPE: "Xorbits/vicuna-13b-v1.5"
        }
    },
    template="vicuna"
)


register_model_group(
    models={
        "XuanYuan-70B": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-70B"
        },
        "XuanYuan-70B-Chat": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-70B-Chat"
        },
        "XuanYuan-70B-int8-Chat": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-70B-Chat-8bit"
        },
        "XuanYuan-70B-int4-Chat": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-70B-Chat-4bit"
        }
    },
    template="xuanyuan"
)


register_model_group(
    models={
        "XVERSE-7B": {
            DownloadSource.DEFAULT: "xverse/XVERSE-7B",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-7B"
        },
        "XVERSE-13B": {
            DownloadSource.DEFAULT: "xverse/XVERSE-13B",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-13B"
        },
        "XVERSE-65B": {
            DownloadSource.DEFAULT: "xverse/XVERSE-65B",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-65B"
        },
        "XVERSE-7B-Chat": {
            DownloadSource.DEFAULT: "xverse/XVERSE-7B-Chat",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-7B-Chat"
        },
        "XVERSE-13B-Chat": {
            DownloadSource.DEFAULT: "xverse/XVERSE-13B-Chat",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-13B-Chat"
        },
        "XVERSE-65B-Chat": {
            DownloadSource.DEFAULT: "xverse/XVERSE-65B-Chat",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-65B-Chat"
        }
    },
    template="xverse"
)


register_model_group(
    models={
        "Yayi-7B": {
            DownloadSource.DEFAULT: "wenge-research/yayi-7b-llama2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/yayi-7b-llama2"
        },
        "Yayi-13B": {
            DownloadSource.DEFAULT: "wenge-research/yayi-13b-llama2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/yayi-13b-llama2"
        }
    },
    template="yayi"
)


register_model_group(
    models={
        "Yi-6B": {
            DownloadSource.DEFAULT: "01-ai/Yi-6B",
            DownloadSource.MODELSCOPE: "01ai/Yi-6B"
        },
        "Yi-34B": {
            DownloadSource.DEFAULT: "01-ai/Yi-34B",
            DownloadSource.MODELSCOPE: "01ai/Yi-34B"
        },
        "Yi-6B-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-6B-Chat",
            DownloadSource.MODELSCOPE: "01ai/Yi-6B-Chat"
        },
        "Yi-34B-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-34B-Chat",
            DownloadSource.MODELSCOPE: "01ai/Yi-34B-Chat"
        },
        "Yi-6B-int8-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-6B-Chat-8bits",
            DownloadSource.MODELSCOPE: "01ai/Yi-6B-Chat-8bits"
        },
        "Yi-34B-int8-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-34B-Chat-8bits",
            DownloadSource.MODELSCOPE: "01ai/Yi-34B-Chat-8bits"
        }
    },
    template="yi"
)


register_model_group(
    models={
        "Zephyr-7B-Alpha-Chat": {
            DownloadSource.DEFAULT: "HuggingFaceH4/zephyr-7b-alpha",
            DownloadSource.MODELSCOPE: "AI-ModelScope/zephyr-7b-alpha"
        },
        "Zephyr-7B-Beta-Chat": {
            DownloadSource.DEFAULT: "HuggingFaceH4/zephyr-7b-beta",
            DownloadSource.MODELSCOPE: "modelscope/zephyr-7b-beta"
        }
    },
    template="zephyr"
)
