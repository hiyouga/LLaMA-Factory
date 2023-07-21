# LLaMA Efficient Tuning

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Efficient-Tuning?style=social)](https://github.com/hiyouga/LLaMA-Efficient-Tuning/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/hiyouga/LLaMA-Efficient-Tuning)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Efficient-Tuning)](https://github.com/hiyouga/LLaMA-Efficient-Tuning/commits/main)
[![PyPI](https://img.shields.io/pypi/v/llmtuner)](https://pypi.org/project/llmtuner/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/hiyouga/LLaMA-Efficient-Tuning/pulls)

👋 加入我们的 [微信群](assets/wechat.jpg).

\[ [English](README.md) | 中文 \]

## 更新日志

[23/07/19] 现在我们在该仓库中提供了对于 **LLaMA-2** 模型的训练支持. 试试 `--model_name_or_path meta-llama/Llama-2-7b-hf` 参数来使用 LLaMA-2 模型. 使用 LLaMA-2-chat 模型时记得使用 `--prompt_template llama2` 参数.

[23/07/18] 我们开发了一个用于训练、评估和推理的 all-in-one Web UI, . 试试 `train_web.py` 在浏览器中微调模型. 感谢 [@KanadeSiina](https://github.com/KanadeSiina) 和 [@codemayq](https://github.com/codemayq) 在项目发展中做出的努力.

[23/07/11] 现在我们在该仓库中提供了对于 **Baichuan-13B** 模型的训练支持. 训练 Baichuan-13B 时使用 `tests/modeling_baichuan.py` , 然后试试 `--model_name_or_path path_to_baichuan_model` 和 `--lora_target W_pack` 参数来训练 Baichuan-13B 模型. 使用 Baichuan-13B-Chat 模型时记得使用 `--prompt_template baichuan` 参数.

[23/07/09] 我们开源了 [FastEdit](https://github.com/hiyouga/FastEdit)⚡🩹，一个简单易用的、能迅速编辑大模型事实记忆的工具包。如果您感兴趣请关注我们的 [FastEdit](https://github.com/hiyouga/FastEdit) 项目。

[23/07/07] 现在我们在该仓库中提供了对于 **InternLM-7B** 模型的训练支持. 试试 `--model_name_or_path internlm/internlm-7b` 参数来使用 InternLM 模型. 使用 InternLM-chat 模型时记得使用 `--prompt_template intern` 参数.

[23/07/05] 现在我们在该仓库中提供了对于 **Falcon-7B/40B** 模型的训练支持. 试试 `--model_name_or_path tiiuae/falcon-7b` 和 `--lora_target query_key_value` 参数来使用 Falcon 模型.

[23/06/29] 我们提供了一个使用 instruction-following 数据集训练聊天模型的 **可复现的示例** , 更多细节请看这里 [Hugging Face Repo](https://huggingface.co/hiyouga/baichuan-7b-sft).

[23/06/22] 我们对齐了[示例 API](src/api_demo.py) 与 [OpenAI API](https://platform.openai.com/docs/api-reference/chat) 的格式，您可以将微调模型接入任意基于 ChatGPT 的应用中.

[23/06/15] 现在我们在该仓库中提供了对于 **Baichuan-7B** 模型的训练支持. 试试 `--model_name_or_path baichuan-inc/Baichuan-7B` 和 `--lora_target W_pack` 参数来使用 Baichuan-7B 模型.

[23/06/03] 现在我们支持了量化训练和推理 (也叫作 **[QLoRA](https://github.com/artidoro/qlora)**). 试试 `--quantization_bit 4/8` 参数来处理量化模型. (实验性功能)

[23/05/31] 现在我们在该仓库中提供了对于 **BLOOM & BLOOMZ** 模型的训练支持. 试试 `--model_name_or_path bigscience/bloomz-7b1-mt` 和 `--lora_target query_key_value` 参数来使用 BLOOMZ 模型.

## 支持的模型

- [LLaMA](https://github.com/facebookresearch/llama) (7B/13B/33B/65B)
- [LLaMA-2](https://huggingface.co/meta-llama) (7B/13B/70B)
- [BLOOM](https://huggingface.co/bigscience/bloom) & [BLOOMZ](https://huggingface.co/bigscience/bloomz) (560M/1.1B/1.7B/3B/7.1B/176B)
- [Falcon](https://huggingface.co/tiiuae/falcon-7b) (7B/40B)
- [Baichuan](https://huggingface.co/baichuan-inc/baichuan-7B) (7B/13B)
- [InternLM](https://github.com/InternLM/InternLM) (7B)

## 支持的训练方法

- [(Continually) pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - Full-parameter tuning
  - Partial-parameter tuning
  - [LoRA](https://arxiv.org/abs/2106.09685)
  - [QLoRA](https://arxiv.org/abs/2305.14314)
- [Supervised fine-tuning](https://arxiv.org/abs/2109.01652)
  - Full-parameter tuning
  - Partial-parameter tuning
  - [LoRA](https://arxiv.org/abs/2106.09685)
  - [QLoRA](https://arxiv.org/abs/2305.14314)
- [RLHF](https://arxiv.org/abs/2203.02155)
  - [LoRA](https://arxiv.org/abs/2106.09685)
  - [QLoRA](https://arxiv.org/abs/2305.14314)

## 提供的训练集

- 对预训练:
  - [Wiki Demo (en)](data/wiki_demo.txt)
- 对监督微调:
  - [Stanford Alpaca (en)](https://github.com/tatsu-lab/stanford_alpaca)
  - [Stanford Alpaca (zh)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
  - [GPT-4 Generated Data (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
  - [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
  - [Self-cognition (zh)](data/self_cognition.json)
  - [ShareGPT (zh)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chinese-instruction-collection)
  - [RefGPT (zh)](https://github.com/sufengniu/RefGPT)
  - [Guanaco Dataset (multilingual)](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
  - [BELLE 2M (zh)](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
  - [BELLE 1M (zh)](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
  - [BELLE 0.5M (zh)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
  - [BELLE Dialogue 0.4M (zh)](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
  - [BELLE School Math 0.25M (zh)](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
  - [BELLE Multiturn Chat 0.8M (zh)](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
  - [Firefly 1.1M (zh)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
  - [CodeAlpaca 20k (en)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
  - [Alpaca CoT (multilingual)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
  - [Web QA (zh)](https://huggingface.co/datasets/suolyer/webqa)
  - [UltraChat (en)](https://github.com/thunlp/UltraChat)
  - [WebNovel (zh)](https://huggingface.co/datasets/zxbsmk/webnovel_cn)
- 对奖励模型:
  - [HH-RLHF (en)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
  - [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
  - [GPT-4 Generated Data (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

更多细节请查看 [data/README.md](data/README_zh.md).

部分数据集使用之前需要确认, 因此推荐使用下面的命令登录您的 Huggingface 账户.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## 软件依赖

- Python 3.8+ 和 PyTorch 1.13.1+
- 🤗Transformers, Datasets, Accelerate, PEFT 和 TRL
- jieba, rouge-chinese 和 nltk (用于评估)
- gradio 和 matplotlib (用于网页端交互)
- uvicorn, fastapi 和 sse-starlette (用于 API)

以及 **强有力的 GPUs**!

如果要在 Windows 平台上开启量化 LoRA (QLoRA) , 需要安装预编译的 `bitsandbytes` 库, 支持 CUDA 11.1 到 12.1.

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.39.1-py3-none-win_amd64.whl
```

## 起步

### 准备数据 (可选)

关于数据集文件的格式，请参考 `data/example_dataset` 文件夹的内容. 构建自定义数据集时, 既可以使用单个 `.json` 文件, 也可以使用一个[数据加载脚本](https://huggingface.co/docs/datasets/dataset_script)和多个文件.

注意：使用自定义数据集时，请更新 `data/dataset_info.json` 文件，该文件的格式请参考 `data/README.md`.

### 环境安装 (可选)

```bash
git clone https://github.com/hiyouga/LLaMA-Efficient-Tuning.git
conda create -n llama_etuning python=3.10
conda activate llama_etuning
cd LLaMA-Efficient-Tuning
pip install -r requirements.txt
```

### All-in-one Web UI

```bash
python src/train_web.py
```

### (Continually) Pre-Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage pt \
    --model_name_or_path path_to_your_model \
    --do_train \
    --dataset wiki_demo \
    --finetuning_type lora \
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

### 监督微调

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
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

### 奖励模型训练

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage rm \
    --model_name_or_path path_to_your_model \
    --do_train \
    --dataset comparison_gpt4_en \
    --finetuning_type lora \
    --output_dir path_to_rm_checkpoint \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
```

### PPO Training (RLHF)

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage ppo \
    --model_name_or_path path_to_your_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
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
    --resume_lora_training False \
    --plot_loss
```

### 分布式微调

```bash
accelerate config # configure the environment
accelerate launch src/train_bash.py # arguments (same as above)
```

<details><summary>使用 DeepSpeed ZeRO-2 全量微调的配置示例</summary>

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 4
  gradient_clipping: 0.5
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
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

### 指标评估（BLEU分数和汉语ROUGE分数）

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_model \
    --do_eval \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_eval_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

我们建议在 4/8-bit 评估中使用 `--per_device_eval_batch_size=1` 和 `--max_target_length 128`.

### 模型预测

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_model \
    --do_predict \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

如果需要预测的样本没有标签，请首先在 `response` 列中填入一些占位符，以免样本在预处理阶段被丢弃。

### API 服务
```bash
python src/api_demo.py \
    --model_name_or_path path_to_your_model \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

访问 `http://localhost:8000/docs` 获取 API 文档.

### 命令行测试

```bash
python src/cli_demo.py \
    --model_name_or_path path_to_your_model \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

### 浏览器测试

```bash
python src/web_demo.py \
    --model_name_or_path path_to_your_model \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

### 导出微调模型

```bash
python src/export_model.py \
    --model_name_or_path path_to_your_model \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_export
```

## 协议

本仓库采用 [Apache-2.0](LICENSE) 协议开源.

请遵循模型许可证使用相应的模型权重:

- [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
- [LLaMA-2](https://ai.meta.com/llama/license/)
- [BLOOM](https://huggingface.co/spaces/bigscience/license)
- [Falcon](LICENSE)
- [baichuan](https://huggingface.co/baichuan-inc/baichuan-7B/resolve/main/baichuan-7B%20%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf)
- [InternLM](https://github.com/InternLM/InternLM#open-source-license)

## 引用

如果您觉得此项目有帮助，请考虑以下列格式引用:

```bibtex
@Misc{llama-efficient-tuning,
  title = {LLaMA Efficient Tuning},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/LLaMA-Efficient-Tuning}},
  year = {2023}
}
```

## 致谢

该 repo 是 [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning) 的兄弟产品. 它们都拥有在大规模语言模型上实现高效调优的相似的代码结构。

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=hiyouga/LLaMA-Efficient-Tuning&type=Date)
