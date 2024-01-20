![# LLaMA Factory](assets/logo.png)

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/hiyouga/LLaMA-Factory)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
[![PyPI](https://img.shields.io/pypi/v/llmtuner)](https://pypi.org/project/llmtuner/)
[![Downloads](https://static.pepy.tech/badge/llmtuner)](https://pypi.org/project/llmtuner/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/hiyouga/LLaMA-Factory/pulls)
[![Discord](https://dcbadge.vercel.app/api/server/rKfvV9r9FK?compact=true&style=flat)](https://discord.gg/rKfvV9r9FK)
[![Spaces](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue)](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
[![Studios](https://img.shields.io/badge/ModelScope-Open%20In%20Studios-blue)](https://modelscope.cn/studios/hiyouga/LLaMA-Board)

👋 Join our [WeChat](assets/wechat.jpg).

\[ English | [中文](README_zh.md) \]

## LLaMA Board: A One-stop Web UI for Getting Started with LLaMA Factory

Preview LLaMA Board at **[🤗 Spaces](https://huggingface.co/spaces/hiyouga/LLaMA-Board)** or **[ModelScope](https://modelscope.cn/studios/hiyouga/LLaMA-Board)**.

Launch LLaMA Board via `CUDA_VISIBLE_DEVICES=0 python src/train_web.py`. (multiple GPUs are not supported yet in this mode)

Here is an example of altering the self-cognition of an instruction-tuned language model within 10 minutes on a single GPU.

https://github.com/hiyouga/LLaMA-Factory/assets/16256802/6ba60acc-e2e2-4bec-b846-2d88920d5ba1

## Table of Contents

- [Benchmark](#benchmark)
- [Changelog](#changelog)
- [Supported Models](#supported-models)
- [Supported Training Approaches](#supported-training-approaches)
- [Provided Datasets](#provided-datasets)
- [Requirement](#requirement)
- [Getting Started](#getting-started)
- [Projects using LLaMA Factory](#projects-using-llama-factory)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Benchmark

Compared to ChatGLM's [P-Tuning](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning), LLaMA-Factory's LoRA tuning offers up to **3.7 times faster** training speed with a better Rouge score on the advertising text generation task. By leveraging 4-bit quantization technique, LLaMA-Factory's QLoRA further improves the efficiency regarding the GPU memory.

![benchmark](assets/benchmark.svg)

<details><summary>Definitions</summary>

- **Training Speed**: the number of training samples processed per second during the training. (bs=4, cutoff_len=1024)
- **Rouge Score**: Rouge-2 score on the development set of the [advertising text generation](https://aclanthology.org/D19-1321.pdf) task. (bs=4, cutoff_len=1024)
- **GPU Memory**: Peak GPU memory usage in 4-bit quantized training. (bs=1, cutoff_len=1024)
- We adopt `pre_seq_len=128` for ChatGLM's P-Tuning and `lora_rank=32` for LLaMA-Factory's LoRA tuning.

</details>

## Changelog

[24/01/18] We supported **agent tuning** for most models, equipping model with tool using abilities by fine-tuning with `--dataset glaive_toolcall`.

[23/12/23] We supported **[unsloth](https://github.com/unslothai/unsloth)**'s implementation to boost LoRA tuning for the LLaMA, Mistral and Yi models. Try `--use_unsloth` argument to activate unsloth patch. It achieves 1.7x speed in our benchmark, check [this page](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-comparison) for details.

[23/12/12] We supported fine-tuning the latest MoE model **[Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)** in our framework. See hardware requirement [here](#hardware-requirement).

<details><summary>Full Changelog</summary>

[23/12/01] We supported downloading pre-trained models and datasets from the **[ModelScope Hub](https://modelscope.cn/models)** for Chinese mainland users. See [this tutorial](#use-modelscope-hub-optional) for usage.

[23/10/21] We supported **[NEFTune](https://arxiv.org/abs/2310.05914)** trick for fine-tuning. Try `--neftune_noise_alpha` argument to activate NEFTune, e.g., `--neftune_noise_alpha 5`.

[23/09/27] We supported **$S^2$-Attn** proposed by [LongLoRA](https://github.com/dvlab-research/LongLoRA) for the LLaMA models. Try `--shift_attn` argument to enable shift short attention.

[23/09/23] We integrated MMLU, C-Eval and CMMLU benchmarks in this repo. See [this example](#evaluation) to evaluate your models.

[23/09/10] We supported **[FlashAttention-2](https://github.com/Dao-AILab/flash-attention)**. Try `--flash_attn` argument to enable FlashAttention-2 if you are using RTX4090, A100 or H100 GPUs.

[23/08/12] We supported **RoPE scaling** to extend the context length of the LLaMA models. Try `--rope_scaling linear` argument in training and `--rope_scaling dynamic` argument at inference to extrapolate the position embeddings.

[23/08/11] We supported **[DPO training](https://arxiv.org/abs/2305.18290)** for instruction-tuned models. See [this example](#dpo-training) to train your models.

[23/07/31] We supported **dataset streaming**. Try `--streaming` and `--max_steps 10000` arguments to load your dataset in streaming mode.

[23/07/29] We released two instruction-tuned 13B models at Hugging Face. See these Hugging Face Repos ([LLaMA-2](https://huggingface.co/hiyouga/Llama-2-Chinese-13b-chat) / [Baichuan](https://huggingface.co/hiyouga/Baichuan-13B-sft)) for details.

[23/07/18] We developed an **all-in-one Web UI** for training, evaluation and inference. Try `train_web.py` to fine-tune models in your Web browser. Thank [@KanadeSiina](https://github.com/KanadeSiina) and [@codemayq](https://github.com/codemayq) for their efforts in the development.

[23/07/09] We released **[FastEdit](https://github.com/hiyouga/FastEdit)** ⚡🩹, an easy-to-use package for editing the factual knowledge of large language models efficiently. Please follow [FastEdit](https://github.com/hiyouga/FastEdit) if you are interested.

[23/06/29] We provided a **reproducible example** of training a chat model using instruction-following datasets, see [Baichuan-7B-sft](https://huggingface.co/hiyouga/Baichuan-7B-sft) for details.

[23/06/22] We aligned the [demo API](src/api_demo.py) with the [OpenAI's](https://platform.openai.com/docs/api-reference/chat) format where you can insert the fine-tuned model in **arbitrary ChatGPT-based applications**.

[23/06/03] We supported quantized training and inference (aka **[QLoRA](https://github.com/artidoro/qlora)**). Try `--quantization_bit 4/8` argument to work with quantized models.

</details>

## Supported Models

| Model                                                    | Model size                  | Default module    | Template  |
| -------------------------------------------------------- | --------------------------- | ----------------- | --------- |
| [Baichuan2](https://huggingface.co/baichuan-inc)         | 7B/13B                      | W_pack            | baichuan2 |
| [BLOOM](https://huggingface.co/bigscience/bloom)         | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value   | -         |
| [BLOOMZ](https://huggingface.co/bigscience/bloomz)       | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value   | -         |
| [ChatGLM3](https://huggingface.co/THUDM/chatglm3-6b)     | 6B                          | query_key_value   | chatglm3  |
| [DeepSeek (MoE)](https://huggingface.co/deepseek-ai)     | 7B/16B/67B                  | q_proj,v_proj     | deepseek  |
| [Falcon](https://huggingface.co/tiiuae)                  | 7B/40B/180B                 | query_key_value   | falcon    |
| [InternLM2](https://huggingface.co/internlm)             | 7B/20B                      | wqkv              | intern2   |
| [LLaMA](https://github.com/facebookresearch/llama)       | 7B/13B/33B/65B              | q_proj,v_proj     | -         |
| [LLaMA-2](https://huggingface.co/meta-llama)             | 7B/13B/70B                  | q_proj,v_proj     | llama2    |
| [Mistral](https://huggingface.co/mistralai)              | 7B                          | q_proj,v_proj     | mistral   |
| [Mixtral](https://huggingface.co/mistralai)              | 8x7B                        | q_proj,v_proj     | mistral   |
| [Phi-1.5/2](https://huggingface.co/microsoft)            | 1.3B/2.7B                   | q_proj,v_proj     | -         |
| [Qwen](https://huggingface.co/Qwen)                      | 1.8B/7B/14B/72B             | c_attn            | qwen      |
| [XVERSE](https://huggingface.co/xverse)                  | 7B/13B/65B                  | q_proj,v_proj     | xverse    |
| [Yi](https://huggingface.co/01-ai)                       | 6B/34B                      | q_proj,v_proj     | yi        |
| [Yuan](https://huggingface.co/IEITYuan)                  | 2B/51B/102B                 | q_proj,v_proj     | yuan      |

> [!NOTE]
> **Default module** is used for the `--lora_target` argument, you can use `--lora_target all` to specify all the available modules.
>
> For the "base" models, the `--template` argument can be chosen from `default`, `alpaca`, `vicuna` etc. But make sure to use the **corresponding template** for the "chat" models.

Please refer to [constants.py](src/llmtuner/extras/constants.py) for a full list of models we supported.

## Supported Training Approaches

| Approach               |   Full-parameter   | Partial-parameter  |       LoRA         |       QLoRA        |
| ---------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| Pre-Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Supervised Fine-Tuning | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Reward Modeling        | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| PPO Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| DPO Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

> [!NOTE]
> Use `--quantization_bit 4` argument to enable QLoRA.

## Provided Datasets

<details><summary>Pre-training datasets</summary>

- [Wiki Demo (en)](data/wiki_demo.txt)
- [RefinedWeb (en)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
- [RedPajama V2 (en)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)
- [Wikipedia (en)](https://huggingface.co/datasets/olm/olm-wikipedia-20221220)
- [Wikipedia (zh)](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- [Pile (en)](https://huggingface.co/datasets/EleutherAI/pile)
- [SkyPile (zh)](https://huggingface.co/datasets/Skywork/SkyPile-150B)
- [The Stack (en)](https://huggingface.co/datasets/bigcode/the-stack)
- [StarCoder (en)](https://huggingface.co/datasets/bigcode/starcoderdata)

</details>

<details><summary>Supervised fine-tuning datasets</summary>

- [Stanford Alpaca (en)](https://github.com/tatsu-lab/stanford_alpaca)
- [Stanford Alpaca (zh)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [GPT-4 Generated Data (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [Self-cognition (zh)](data/self_cognition.json)
- [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [ShareGPT (zh)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chinese-instruction-collection)
- [Guanaco Dataset (multilingual)](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
- [BELLE 2M (zh)](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
- [BELLE 1M (zh)](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- [BELLE 0.5M (zh)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- [BELLE Dialogue 0.4M (zh)](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
- [BELLE School Math 0.25M (zh)](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
- [BELLE Multiturn Chat 0.8M (zh)](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
- [UltraChat (en)](https://github.com/thunlp/UltraChat)
- [LIMA (en)](https://huggingface.co/datasets/GAIR/lima)
- [OpenPlatypus (en)](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
- [CodeAlpaca 20k (en)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
- [Alpaca CoT (multilingual)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
- [OpenOrca (en)](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [MathInstruct (en)](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
- [Firefly 1.1M (zh)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- [Web QA (zh)](https://huggingface.co/datasets/suolyer/webqa)
- [WebNovel (zh)](https://huggingface.co/datasets/zxbsmk/webnovel_cn)
- [Nectar (en)](https://huggingface.co/datasets/berkeley-nest/Nectar)
- [deepctrl (en&zh)](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
- [Ad Gen (zh)](https://huggingface.co/datasets/HasturOfficial/adgen)
- [ShareGPT Hyperfiltered (en)](https://huggingface.co/datasets/totally-not-an-llm/sharegpt-hyperfiltered-3k)
- [ShareGPT4 (en&zh)](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)
- [UltraChat 200k (en)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [AgentInstruct (en)](https://huggingface.co/datasets/THUDM/AgentInstruct)
- [LMSYS Chat 1M (en)](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- [Evol Instruct V2 (en)](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)
- [Glaive Function Calling V2 (en)](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

</details>

<details><summary>Preference datasets</summary>

- [HH-RLHF (en)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [GPT-4 Generated Data (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [Nectar (en)](https://huggingface.co/datasets/berkeley-nest/Nectar)

</details>

Please refer to [data/README.md](data/README.md) for details.

Some datasets require confirmation before using them, so we recommend logging in with your Hugging Face account using these commands.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## Requirement

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers, Datasets, Accelerate, PEFT and TRL
- sentencepiece, protobuf and tiktoken
- jieba, rouge-chinese and nltk (used at evaluation and predict)
- gradio and matplotlib (used in web UI)
- uvicorn, fastapi and sse-starlette (used in API)

### Hardware Requirement

| Method | Bits |   7B  |  13B  |  30B  |   65B  |   8x7B |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ |
| Full   |  16  | 160GB | 320GB | 600GB | 1200GB |  900GB |
| Freeze |  16  |  20GB |  40GB | 120GB |  240GB |  200GB |
| LoRA   |  16  |  16GB |  32GB |  80GB |  160GB |  120GB |
| QLoRA  |   8  |  10GB |  16GB |  40GB |   80GB |   80GB |
| QLoRA  |   4  |   6GB |  12GB |  24GB |   48GB |   32GB |

## Getting Started

### Data Preparation (optional)

Please refer to [data/README.md](data/README.md) for checking the details about the format of dataset files. You can either use a single `.json` file or a [dataset loading script](https://huggingface.co/docs/datasets/dataset_script) with multiple files to create a custom dataset.

> [!NOTE]
> Please update `data/dataset_info.json` to use your custom dataset. About the format of this file, please refer to `data/README.md`.

### Dependence Installation (optional)

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
conda create -n llama_factory python=3.10
conda activate llama_factory
cd LLaMA-Factory
pip install -r requirements.txt
```

If you want to enable the quantized LoRA (QLoRA) on the Windows platform, you will be required to install a pre-built version of `bitsandbytes` library, which supports CUDA 11.1 to 12.1.

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.39.1-py3-none-win_amd64.whl
```

### Use ModelScope Hub (optional)

If you have trouble with downloading models and datasets from Hugging Face, you can use LLaMA-Factory together with ModelScope in the following manner.

```bash
export USE_MODELSCOPE_HUB=1 # `set USE_MODELSCOPE_HUB=1` for Windows
```

Then you can train the corresponding model by specifying a model ID of the ModelScope Hub. (find a full list of model IDs at [ModelScope Hub](https://modelscope.cn/models))

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --model_name_or_path modelscope/Llama-2-7b-ms \
    ... # arguments (same as above)
```

LLaMA Board also supports using the models and datasets on the ModelScope Hub.

```bash
CUDA_VISIBLE_DEVICES=0 USE_MODELSCOPE_HUB=1 python src/train_web.py
```

### Train on a single GPU

> [!IMPORTANT]
> If you want to train models on multiple GPUs, please refer to [Distributed Training](#distributed-training).

#### Pre-Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage pt \
    --do_train \
    --model_name_or_path path_to_llama_model \
    --dataset wiki_demo \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_pt_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

#### Supervised Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path path_to_llama_model \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

#### Reward Modeling

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_sft_checkpoint \
    --create_new_adapter \
    --dataset comparison_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_rm_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
```

#### PPO Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage ppo \
    --do_train \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_sft_checkpoint \
    --create_new_adapter \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --reward_model path_to_rm_checkpoint \
    --output_dir path_to_ppo_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --top_k 0 \
    --top_p 0.9 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
```

> [!WARNING]
> Use `--per_device_train_batch_size=1` for LLaMA-2 models in fp16 PPO training.

#### DPO Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_sft_checkpoint \
    --create_new_adapter \
    --dataset comparison_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_dpo_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
```

### Distributed Training

#### Use Huggingface Accelerate

```bash
accelerate config # configure the environment
accelerate launch src/train_bash.py # arguments (same as above)
```

<details><summary>Example config for LoRA training</summary>

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</details>

#### Use DeepSpeed

```bash
deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    ... # arguments (same as above)
```

<details><summary>Example config for full-parameter training with DeepSpeed ZeRO-2</summary>

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": false,
    "contiguous_gradients": true
  }
}
```

</details>

### Merge LoRA weights and export model

```bash
python src/export_model.py \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_checkpoint \
    --template default \
    --finetuning_type lora \
    --export_dir path_to_export \
    --export_size 2 \
    --export_legacy_format False
```

> [!WARNING]
> Merging LoRA weights into a quantized model is not supported.

> [!TIP]
> Use `--export_quantization_bit 4` and `--export_quantization_dataset data/c4_demo.json` to quantize the model after merging the LoRA weights.

### API Demo

```bash
python src/api_demo.py \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_checkpoint \
    --template default \
    --finetuning_type lora
```

> [!TIP]
> Visit `http://localhost:8000/docs` for API documentation.

### CLI Demo

```bash
python src/cli_demo.py \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_checkpoint \
    --template default \
    --finetuning_type lora
```

### Web Demo

```bash
python src/web_demo.py \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_checkpoint \
    --template default \
    --finetuning_type lora
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_checkpoint \
    --template vanilla \
    --finetuning_type lora \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 4
```

### Predict

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_checkpoint \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate \
    --fp16
```

> [!WARNING]
> Use `--per_device_train_batch_size=1` for LLaMA-2 models in fp16 predict.

> [!TIP]
> We recommend using `--per_device_eval_batch_size=1` and `--max_target_length 128` at 4/8-bit predict.

## Projects using LLaMA Factory

- **[StarWhisper](https://github.com/Yu-Yang-Li/StarWhisper)**: A large language model for Astronomy, based on ChatGLM2-6B and Qwen-14B.
- **[DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM)**: A large language model specialized in Chinese legal domain, based on Baichuan-13B, is capable of retrieving and reasoning on legal knowledge.
- **[Sunsimiao](https://github.com/thomas-yanxin/Sunsimiao)**: A large language model specialized in Chinese medical domain, based on Baichuan-7B and ChatGLM-6B.
- **[CareGPT](https://github.com/WangRongsheng/CareGPT)**: A series of large language models for Chinese medical domain, based on LLaMA2-7B and Baichuan-13B.
- **[MachineMindset](https://github.com/PKU-YuanGroup/Machine-Mindset/)**: A series of MBTI Personality large language models, capable of giving any LLM 16 different personality types based on different datasets and training methods.

> [!TIP]
> If you have a project that should be incorporated, please contact via email or create a pull request.

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

Please follow the model licenses to use the corresponding model weights: [Baichuan2](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/Community%20License%20for%20Baichuan%202%20Model.pdf) / [BLOOM](https://huggingface.co/spaces/bigscience/license) / [ChatGLM3](https://github.com/THUDM/ChatGLM3/blob/main/MODEL_LICENSE) / [DeepSeek](https://github.com/deepseek-ai/DeepSeek-LLM/blob/main/LICENSE-MODEL) / [Falcon](https://huggingface.co/tiiuae/falcon-180B/blob/main/LICENSE.txt) / [InternLM2](https://github.com/InternLM/InternLM#license) / [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) / [LLaMA-2](https://ai.meta.com/llama/license/) / [Mistral](LICENSE) / [Phi-1.5/2](https://huggingface.co/microsoft/phi-1_5/resolve/main/Research%20License.docx) / [Qwen](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT) / [XVERSE](https://github.com/xverse-ai/XVERSE-13B/blob/main/MODEL_LICENSE.pdf) / [Yi](https://huggingface.co/01-ai/Yi-6B/blob/main/LICENSE) / [Yuan](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/LICENSE-Yuan)

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@Misc{llama-factory,
  title = {LLaMA Factory},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/LLaMA-Factory}},
  year = {2023}
}
```

## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [QLoRA](https://github.com/artidoro/qlora) and [FastChat](https://github.com/lm-sys/FastChat). Thanks for their wonderful works.

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=hiyouga/LLaMA-Factory&type=Date)
