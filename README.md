# LLaMA Efficient Tuning

![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Efficient-Tuning?style=social)
![GitHub Code License](https://img.shields.io/github/license/hiyouga/LLaMA-Efficient-Tuning)
![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Efficient-Tuning)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)

👋 Join our [WeChat](assets/wechat.jpg).

## Changelog

[23/07/05] Now we support training the Falcon-7B/40B models in this repo. Try `--model_name_or_path tiiuae/falcon-7b` and `--lora_target query_key_value` arguments to use the Falcon model.

[23/06/29] We provide a reproducible example of training a chat model using instruction-following datasets, see this [HuggingFace Repo](https://huggingface.co/hiyouga/baichuan-7b-sft) for details.

[23/06/22] Now we align the [demo API](src/api_demo.py) with the [OpenAI's](https://platform.openai.com/docs/api-reference/chat) format where you can insert the fine-tuned model in arbitrary ChatGPT-based applications.

[23/06/15] Now we support training the baichuan-7B model in this repo. Try `--model_name_or_path baichuan-inc/baichuan-7B` and `--lora_target W_pack` arguments to use the baichuan-7B model. If you want to train with RTX3090, use `git checkout baichuan-7b-rtx3090` to switch to the `baichuan-7b-rtx3090` branch and try the `--baichuan_rtx_gpu true` argument. (Other RTX series GPUs can also be tried)

[23/06/03] Now we support quantized training and inference (aka [QLoRA](https://github.com/artidoro/qlora)). Try `--quantization_bit 4/8` argument to work with quantized model. (experimental feature)

[23/05/31] Now we support training the BLOOM & BLOOMZ models in this repo. Try `--model_name_or_path bigscience/bloomz-7b1-mt` and `--lora_target query_key_value` arguments to use the BLOOMZ model.

## Supported Models

- [LLaMA](https://github.com/facebookresearch/llama) (7B/13B/33B/65B)
- [BLOOM](https://huggingface.co/bigscience/bloom) & [BLOOMZ](https://huggingface.co/bigscience/bloomz) (560M/1.1B/1.7B/3B/7.1B/176B)
- [Falcon](https://huggingface.co/tiiuae/falcon-7b) (7B/40B)
- [baichuan](https://huggingface.co/baichuan-inc/baichuan-7B) (7B)

## Supported Training Approaches

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

## Provided Datasets

- For pre-training:
  - [Wiki Demo](data/wiki_demo.txt)
- For supervised fine-tuning:
  - [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
  - [Stanford Alpaca (Chinese)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
  - [GPT-4 Generated Data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
  - [BELLE 2M](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
  - [BELLE 1M](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
  - [BELLE 0.5M](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
  - [BELLE Dialogue 0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
  - [BELLE School Math 0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
  - [BELLE Multiturn Chat 0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
  - [Guanaco Dataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
  - [Firefly 1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
  - [CodeAlpaca 20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
  - [Alpaca CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
  - [Web QA (Chinese)](https://huggingface.co/datasets/suolyer/webqa)
  - [UltraChat](https://github.com/thunlp/UltraChat)
  - [Open Assistant](https://huggingface.co/datasets/OpenAssistant/oasst1)
  - [Open Assistant (Chinese)](https://huggingface.co/datasets/OpenAssistant/oasst1)
- For reward model training:
  - [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
  - [Open Assistant](https://huggingface.co/datasets/OpenAssistant/oasst1)
  - [Open Assistant (Chinese)](https://huggingface.co/datasets/OpenAssistant/oasst1)
  - [GPT-4 Generated Data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
  - [GPT-4 Generated Data (Chinese)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

Please refer to [data/README.md](data/README.md) for details.

Some datasets require confirmation before using them, so we recommend logging in with your HuggingFace account using these commands.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## Requirement

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers, Datasets, Accelerate, PEFT and TRL
- jieba, rouge_chinese and nltk (used at evaluation)
- gradio and mdtex2html (used in web_demo.py)
- uvicorn and fastapi (used in api_demo.py)

And **powerful GPUs**!

If you want to enable quantized LoRA (QLoRA) on the Windows platform, you should install a pre-built version of `bitsandbytes` library, which supports CUDA 11.1 to 12.1.

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.39.1-py3-none-win_amd64.whl
```

## Getting Started

### Data Preparation (optional)

Please refer to `data/example_dataset` for checking the details about the format of dataset files. You can either use a single `.json` file or a [dataset loading script](https://huggingface.co/docs/datasets/dataset_script) with multiple files to create a custom dataset.

Note: please update `data/dataset_info.json` to use your custom dataset. About the format of this file, please refer to `data/README.md`.

### Dependence Installation (optional)

```bash
git clone https://github.com/hiyouga/LLaMA-Efficient-Tuning.git
conda create -n llama_etuning python=3.10
conda activate llama_etuning
cd LLaMA-Efficient-Tuning
pip install -r requirements.txt
```

### LLaMA Weights Preparation (optional)

1. Download the weights of the LLaMA models.
2. Convert them to HF format using the following command.

```bash
python -m transformers.models.llama.convert_llama_weights_to_hf \
    --input_dir path_to_llama_weights --model_size 7B --output_dir path_to_llama_model
```

### (Continually) Pre-Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_pt.py \
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

### Supervised Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
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

### Reward Model Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_rm.py \
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
CUDA_VISIBLE_DEVICES=0 python src/train_ppo.py \
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

### Distributed Training

```bash
accelerate config # configure the environment
accelerate launch src/train_XX.py # arguments (same as above)
```

<details><summary>Example configuration for full-tuning with DeepSpeed ZeRO-2</summary>

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

### Evaluation (BLEU and ROUGE_CHINESE)

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
    --model_name_or_path path_to_your_model \
    --do_eval \
    --dataset alpaca_gpt4_en \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_eval_result \
    --per_device_eval_batch_size 8 \
    --max_samples 50 \
    --predict_with_generate
```

We recommend using `--per_device_eval_batch_size=1` and `--max_target_length 128` at 4/8-bit evaluation.

### API / CLI / Web Demo

```bash
python src/xxx_demo.py \
    --model_name_or_path path_to_your_model \
    --checkpoint_dir path_to_checkpoint
```

### Export model

```bash
python src/export_model.py \
    --model_name_or_path path_to_your_model \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_export
```

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

Please follow the [Model Card](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) to use the LLaMA models.

Please follow the [RAIL License](https://huggingface.co/spaces/bigscience/license) to use the BLOOM & BLOOMZ models.

Please follow the [Apache-2.0 License](LICENSE) to use the Falcon models.

Please follow the [baichuan-7B License](https://huggingface.co/baichuan-inc/baichuan-7B/resolve/main/baichuan-7B%20%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf) to use the baichuan-7B model.

## Citation

If this work is helpful, please cite as:

```bibtex
@Misc{llama-efficient-tuning,
  title = {LLaMA Efficient Tuning},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/LLaMA-Efficient-Tuning}},
  year = {2023}
}
```

## Acknowledgement

This repo is a sibling of [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning). They share a similar code structure of efficient tuning on large language models.

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=hiyouga/LLaMA-Efficient-Tuning&type=Date)
