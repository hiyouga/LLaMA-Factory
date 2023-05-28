Data format in `dataset_info.json`:
```json
"dataset_name": {
    "hf_hub_url": "the name of the dataset repository on the HuggingFace hub. (if specified, ignore below 3 arguments)",
    "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore below 2 arguments)",
    "file_name": "the name of the dataset file in the this directory. (required if above are not specified)",
    "file_sha1": "the SHA-1 hash value of the dataset file. (optional)",
    "columns": {
        "prompt": "the name of the column in the datasets containing the prompts. (default: instruction)",
        "query": "the name of the column in the datasets containing the queries. (default: input)",
        "response": "the name of the column in the datasets containing the responses. (default: output)",
        "history": "the name of the column in the datasets containing the history of chat. (default: None)"
    }
}
```

`dataset_info.json` 中的数据集定义格式：
```json
"数据集名称": {
    "hf_hub_url": "HuggingFace上的项目地址（若指定，则忽略下列三个参数）",
    "script_url": "包含数据加载脚本的本地文件夹名称（若指定，则忽略下列两个参数）",
    "file_name": "该目录下数据集文件的名称（若上述参数未指定，则此项必需）",
    "file_sha1": "数据集文件的SHA-1哈希值（可选）",
    "columns": {
        "prompt": "数据集代表提示词的表头名称（默认：instruction）",
        "query": "数据集代表请求的表头名称（默认：input）",
        "response": "数据集代表回答的表头名称（默认：output）",
        "history": "数据集代表历史对话的表头名称（默认：None）"
    }
}
```

部分预置数据集简介：

| 数据集名称 | 规模 | 描述 |
| --- | --- | --- |
| [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 52k | 斯坦福大学开源的 Alpaca 数据集，训练了 Alpaca 这类早期基于 LLaMA 的模型 |
| [Stanford Alpaca (Chinese)](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | 51k | 使用 ChatGPT 翻译的 Alpaca 数据集 |
| [GPT-4 Generated Data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | 100k+ | 基于 GPT-4 的 self-instruction 数据集 |
| [BELLE 2M](https://huggingface.co/datasets/BelleGroup/train_2M_CN) | 2m | 包含约 200 万条由 [BELLE](https://github.com/LianjiaTech/BELLE) 项目生成的中文指令数据 |
| [BELLE 1M](https://huggingface.co/datasets/BelleGroup/train_1M_CN) | 1m | 包含约 100 万条由 [BELLE](https://github.com/LianjiaTech/BELLE) 项目生成的中文指令数据 |
| [BELLE 0.5M](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) | 500k  | 包含约 50 万条由 [BELLE](https://github.com/LianjiaTech/BELLE) 项目生成的中文指令数据 |
| [BELLE Dialogue 0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M) | 400k | 包含约 40 万条由 [BELLE](https://github.com/LianjiaTech/BELLE) 项目生成的个性化角色对话数据，包含角色介绍 |
| [BELLE School Math 0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) | 250k  | 包含约 25 万条由 [BELLE](https://github.com/LianjiaTech/BELLE) 项目生成的中文数学题数据，包含解题过程 |
| [BELLE Multiturn Chat 0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) | 800k | 包含约 80 万条由 [BELLE](https://github.com/LianjiaTech/BELLE) 项目生成的用户与助手的多轮对话 |
| [Guanaco Dataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) | 100k+ | 包含日文、简繁体中文、英文等多类数据，数据集原用于 Guanaco 模型训练 |
| [Firefly 1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) | 1.1M  | 中文对话大模型 firefly（流萤）的中文数据集，包含多个 NLP 任务 |
| [CodeAlpaca 20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | 20k | 英文代码生成任务数据集 |
| [Alpaca CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT) | 6M | 用于微调的指令数据集集合 |
| [Web QA](https://huggingface.co/datasets/suolyer/webqa) | 36k | 百度知道汇集的中文问答数据集 |
| [UltraChat](https://github.com/thunlp/UltraChat) | 1.57M | 清华 NLP 发布的大规模多轮对话数据集 |

注：BELLE 数据集是由 ChatGPT 产生的数据集，不保证数据准确性，所有类 GPT 模型产生的 self-instruction 数据集均不能保证其准确性。
