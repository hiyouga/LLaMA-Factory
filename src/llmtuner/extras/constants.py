IGNORE_INDEX = -100

LOG_FILE_NAME = "trainer_log.jsonl"

LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp", "ln_1", "ln_2"]

METHODS = ["full", "freeze", "lora"]

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "Pre-Training": "pt"
}

SUPPORTED_MODELS = {
    "LLaMA-7B": "huggyllama/llama-7b",
    "LLaMA-13B": "huggyllama/llama-13b",
    "LLaMA-30B": "huggyllama/llama-30b",
    "LLaMA-65B": "huggyllama/llama-65b",
    "LLaMA2-7B": "meta-llama/Llama-2-7b-hf",
    "LLaMA2-13B": "meta-llama/Llama-2-13b-hf",
    "LLaMA2-70B": "meta-llama/Llama-2-70b-hf",
    "LLaMA2-7B-Chat": "meta-llama/Llama-2-7b-chat-hf",
    "LLaMA2-13B-Chat": "meta-llama/Llama-2-13b-chat-hf",
    "LLaMA2-70B-Chat": "meta-llama/Llama-2-70b-chat-hf",
    "ChineseLLaMA2-7B": "ziqingyang/chinese-llama-2-7b",
    "ChineseLLaMA2-13B": "ziqingyang/chinese-llama-2-13b",
    "ChineseLLaMA2-7B-Chat": "ziqingyang/chinese-alpaca-2-7b",
    "ChineseLLaMA2-13B-Chat": "ziqingyang/chinese-alpaca-2-13b",
    "BLOOM-560M": "bigscience/bloom-560m",
    "BLOOM-3B": "bigscience/bloom-3b",
    "BLOOM-7B1": "bigscience/bloom-7b1",
    "BLOOMZ-560M": "bigscience/bloomz-560m",
    "BLOOMZ-3B": "bigscience/bloomz-3b",
    "BLOOMZ-7B1-mt": "bigscience/bloomz-7b1-mt",
    "Falcon-7B": "tiiuae/falcon-7b",
    "Falcon-40B": "tiiuae/falcon-40b",
    "Falcon-7B-Chat": "tiiuae/falcon-7b-instruct",
    "Falcon-40B-Chat": "tiiuae/falcon-40b-instruct",
    "Baichuan-7B": "baichuan-inc/Baichuan-7B",
    "Baichuan-13B": "baichuan-inc/Baichuan-13B-Base",
    "Baichuan-13B-Chat": "baichuan-inc/Baichuan-13B-Chat",
    "Baichuan2-7B": "baichuan-inc/Baichuan2-7B-Base",
    "Baichuan2-13B": "baichuan-inc/Baichuan2-13B-Base",
    "Baichuan2-7B-Chat": "baichuan-inc/Baichuan2-7B-Chat",
    "Baichuan2-13B-Chat": "baichuan-inc/Baichuan2-13B-Chat",
    "InternLM-7B": "internlm/internlm-7b",
    "InternLM-20B": "internlm/internlm-20b",
    "InternLM-7B-Chat": "internlm/internlm-chat-7b",
    "InternLM-20B-Chat": "internlm/internlm-chat-20b",
    "Qwen-7B": "Qwen/Qwen-7B",
    "Qwen-14B": "Qwen/Qwen-14B",
    "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
    "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat",
    "XVERSE-13B": "xverse/XVERSE-13B",
    "XVERSE-13B-Chat": "xverse/XVERSE-13B-Chat",
    "ChatGLM2-6B-Chat": "THUDM/chatglm2-6b",
    "ChatGLM3-6B-Base": "THUDM/chatglm3-6b-base",
    "ChatGLM3-6B-Chat": "THUDM/chatglm3-6b",
    "Phi1.5-1.3B": "microsoft/phi-1_5"
}

DEFAULT_MODULE = {
    "LLaMA": "q_proj,v_proj",
    "LLaMA2": "q_proj,v_proj",
    "ChineseLLaMA2": "q_proj,v_proj",
    "BLOOM": "query_key_value",
    "BLOOMZ": "query_key_value",
    "Falcon": "query_key_value",
    "Baichuan": "W_pack",
    "Baichuan2": "W_pack",
    "InternLM": "q_proj,v_proj",
    "Qwen": "c_attn",
    "XVERSE": "q_proj,v_proj",
    "ChatGLM2": "query_key_value",
    "ChatGLM3": "query_key_value",
    "Phi1.5": "Wqkv"
}

DEFAULT_TEMPLATE = {
    "LLaMA2": "llama2",
    "ChineseLLaMA2": "llama2_zh",
    "Baichuan": "baichuan",
    "Baichuan2": "baichuan2",
    "InternLM": "intern",
    "Qwen": "chatml",
    "XVERSE": "xverse",
    "ChatGLM2": "chatglm2",
    "ChatGLM3": "chatglm3"
}
