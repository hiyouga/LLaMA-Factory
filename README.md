# LLaMA Factory: Training and Evaluating Large Language Models with Minimal Effort

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/hiyouga/LLaMA-Factory)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
[![PyPI](https://img.shields.io/pypi/v/llmtuner)](https://pypi.org/project/llmtuner/)
[![Downloads](https://static.pepy.tech/badge/llmtuner)](https://pypi.org/project/llmtuner/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/hiyouga/LLaMA-Factory/pulls)
[![Discord](https://dcbadge.vercel.app/api/server/e73gccsSd?compact=true&style=flat)](https://discord.gg/e73gccsSd)

👋 Join our [WeChat](assets/wechat.jpg).

\[ English | [中文](README_zh.md) \]

## LLaMA Board: A One-stop Web UI for Getting Started with LLaMA Factory

Launch **LLaMA Board** via `CUDA_VISIBLE_DEVICES=0 python src/train_web.py`. (multiple GPUs are not supported yet)

Here is an example of altering the self-cognition of an instruction-tuned language model within 10 minutes on a single GPU.

https://github.com/hiyouga/LLaMA-Factory/assets/16256802/6ba60acc-e2e2-4bec-b846-2d88920d5ba1

## Changelog

[23/09/27] We supported **$S^2$-Attn** proposed by [LongLoRA](https://github.com/dvlab-research/LongLoRA) for the LLaMA models. Try `--shift_attn` argument to enable shift short attention.

[23/09/23] We integrated MMLU, C-Eval and CMMLU benchmarks in this repo. See [this example](#evaluation) to evaluate your models.

[23/09/10] We supported using **[FlashAttention-2](https://github.com/Dao-AILab/flash-attention)** for the LLaMA models. Try `--flash_attn` argument to enable FlashAttention-2 if you are using RTX4090, A100 or H100 GPUs.

[23/08/12] We supported **RoPE scaling** to extend the context length of the LLaMA models. Try `--rope_scaling linear` argument in training and `--rope_scaling dynamic` argument at inference to extrapolate the position embeddings.

[23/08/11] We supported **[DPO training](https://arxiv.org/abs/2305.18290)** for instruction-tuned models. See [this example](#dpo-training) to train your models.

[23/07/31] We supported **dataset streaming**. Try `--streaming` and `--max_steps 10000` arguments to load your dataset in streaming mode.

[23/07/29] We released two instruction-tuned 13B models at Hugging Face. See these Hugging Face Repos ([LLaMA-2](https://huggingface.co/hiyouga/Llama-2-Chinese-13b-chat) / [Baichuan](https://huggingface.co/hiyouga/Baichuan-13B-sft)) for details.

[23/07/18] We developed an **all-in-one Web UI** for training, evaluation and inference. Try `train_web.py` to fine-tune models in your Web browser. Thank [@KanadeSiina](https://github.com/KanadeSiina) and [@codemayq](https://github.com/codemayq) for their efforts in the development.

[23/07/09] We released **[FastEdit](https://github.com/hiyouga/FastEdit)** ⚡🩹, an easy-to-use package for editing the factual knowledge of large language models efficiently. Please follow [FastEdit](https://github.com/hiyouga/FastEdit) if you are interested.

[23/06/29] We provided a **reproducible example** of training a chat model using instruction-following datasets, see [Baichuan-7B-sft](https://huggingface.co/hiyouga/Baichuan-7B-sft) for details.

[23/06/22] We aligned the [demo API](src/api_demo.py) with the [OpenAI's](https://platform.openai.com/docs/api-reference/chat) format where you can insert the fine-tuned model in **arbitrary ChatGPT-based applications**.

[23/06/03] We supported quantized training and inference (aka **[QLoRA](https://github.com/artidoro/qlora)**). Try `--quantization_bit 4/8` argument to work with quantized models.

## Supported Models

| Model                                                    | Model size                  | Default module    | Template  |
| -------------------------------------------------------- | --------------------------- | ----------------- | --------- |
| [LLaMA](https://github.com/facebookresearch/llama)       | 7B/13B/33B/65B              | q_proj,v_proj     | -         |
| [LLaMA-2](https://huggingface.co/meta-llama)             | 7B/13B/70B                  | q_proj,v_proj     | llama2    |
| [BLOOM](https://huggingface.co/bigscience/bloom)         | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value   | -         |
| [BLOOMZ](https://huggingface.co/bigscience/bloomz)       | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value   | -         |
| [Falcon](https://huggingface.co/tiiuae/falcon-7b)        | 7B/40B                      | query_key_value   | -         |
| [Baichuan](https://github.com/baichuan-inc/Baichuan-13B) | 7B/13B                      | W_pack            | baichuan  |
| [Baichuan2](https://github.com/baichuan-inc/Baichuan2)   | 7B/13B                      | W_pack            | baichuan2 |
| [InternLM](https://github.com/InternLM/InternLM)         | 7B/20B                      | q_proj,v_proj     | intern    |
| [Qwen](https://github.com/QwenLM/Qwen-7B)                | 7B/14B                      | c_attn            | chatml    |
| [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)         | 6B                          | query_key_value   | chatglm2  |
| [Phi-1.5](https://huggingface.co/microsoft/phi-1_5)      | 1.3B                        | Wqkv              | -         |

> [!NOTE]
> **Default module** is used for the `--lora_target` argument, you can use `--lora_target all` to specify all the available modules.
>
> For the "base" models, the `--template` argument can be chosen from `default`, `alpaca`, `vicuna` etc. But make sure to use the **corresponding template** for the "chat" models.
>
> Please refer to [template.py](src/llmtuner/extras/template.py) for a full list of models we supported.

## Supported Training Approaches

| Approach               |   Full-parameter   | Partial-parameter  |       LoRA         |       QLoRA        |
| ---------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| Pre-Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Supervised Fine-Tuning | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Reward Modeling        |                    |                    | :white_check_mark: | :white_check_mark: |
| PPO Training           |                    |                    | :white_check_mark: | :white_check_mark: |
| DPO Training           | :white_check_mark: |                    | :white_check_mark: | :white_check_mark: |

> [!NOTE]
> Use `--quantization_bit 4/8` argument to enable QLoRA.

## Provided Datasets

- For pre-training:
  - [Wiki Demo (en)](data/wiki_demo.txt)
  - [RefinedWeb (en)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
  - [StarCoder (en)](https://huggingface.co/datasets/bigcode/starcoderdata)
  - [Wikipedia (en)](https://huggingface.co/datasets/olm/olm-wikipedia-20221220)
  - [Wikipedia (zh)](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- For supervised fine-tuning:
  - [Stanford Alpaca (en)](https://github.com/tatsu-lab/stanford_alpaca)
  - [Stanford Alpaca (zh)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
  - [GPT-4 Generated Data (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
  - [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
  - [Self-cognition (zh)](data/self_cognition.json)
  - [ShareGPT (zh)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chinese-instruction-collection)
  - [Guanaco Dataset (multilingual)](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
  - [BELLE 2M (zh)](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
  - [BELLE 1M (zh)](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
  - [BELLE 0.5M (zh)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
  - [BELLE Dialogue 0.4M (zh)](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
  - [BELLE School Math 0.25M (zh)](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
  - [BELLE Multiturn Chat 0.8M (zh)](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
  - [LIMA (en)](https://huggingface.co/datasets/GAIR/lima)
  - [CodeAlpaca 20k (en)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
  - [Alpaca CoT (multilingual)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
  - [MathInstruct (en)](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
  - [Firefly 1.1M (zh)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
  - [Web QA (zh)](https://huggingface.co/datasets/suolyer/webqa)
  - [UltraChat (en)](https://github.com/thunlp/UltraChat)
  - [WebNovel (zh)](https://huggingface.co/datasets/zxbsmk/webnovel_cn)
  - [Ad Gen (zh)](https://huggingface.co/datasets/HasturOfficial/adgen)
- For reward modeling or DPO training:
  - [HH-RLHF (en)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
  - [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
  - [GPT-4 Generated Data (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

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
- fire, jieba, rouge-chinese and nltk (used at evaluation and predict)
- gradio and matplotlib (used in web_demo.py)
- uvicorn, fastapi and sse-starlette (used in api_demo.py)

And **powerful GPUs**!

## Getting Started

### Data Preparation (optional)

Please refer to `data/example_dataset` for checking the details about the format of dataset files. You can either use a single `.json` file or a [dataset loading script](https://huggingface.co/docs/datasets/dataset_script) with multiple files to create a custom dataset.

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

### Train on a single GPU

> [!IMPORTANT]
> If you want to train models on multiple GPUs, please refer to [Distributed Training](#distributed-training).

#### Pre-Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage pt \
    --model_name_or_path path_to_llama_model \
    --do_train \
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
    --model_name_or_path path_to_llama_model \
    --do_train \
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
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset comparison_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --checkpoint_dir path_to_sft_checkpoint \
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
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --checkpoint_dir path_to_sft_checkpoint \
    --reward_model path_to_rm_checkpoint \
    --output_dir path_to_ppo_checkpoint \
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

#### DPO Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage dpo \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset comparison_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --checkpoint_dir path_to_sft_checkpoint \
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

### Export model

```bash
python src/export_model.py \
    --model_name_or_path path_to_llama_model \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --export_dir path_to_export
```

### API Demo

```bash
python src/api_demo.py \
    --model_name_or_path path_to_llama_model \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

> [!NOTE]
> Visit `http://localhost:8000/docs` for API documentation.

### CLI Demo

```bash
python src/cli_demo.py \
    --model_name_or_path path_to_llama_model \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

### Web Demo

```bash
python src/web_demo.py \
    --model_name_or_path path_to_llama_model \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name_or_path path_to_llama_model \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --template vanilla \
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
    --model_name_or_path path_to_llama_model \
    --do_predict \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

> [!NOTE]
> We recommend using `--per_device_eval_batch_size=1` and `--max_target_length 128` at 4/8-bit predict.

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

Please follow the model licenses to use the corresponding model weights: [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) / [LLaMA-2](https://ai.meta.com/llama/license/) / [BLOOM](https://huggingface.co/spaces/bigscience/license) / [Falcon](LICENSE) / [Baichuan](https://huggingface.co/baichuan-inc/baichuan-7B/resolve/main/baichuan-7B%20%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf) / [Baichuan2](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/Baichuan%202%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf) / [InternLM](https://github.com/InternLM/InternLM#open-source-license) / [Qwen](https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/LICENSE) / [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B/blob/main/MODEL_LICENSE) / [Phi-1.5](https://huggingface.co/microsoft/phi-1_5/resolve/main/Research%20License.docx)

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
