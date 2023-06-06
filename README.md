# LLaMA Efficient Tuning

![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Efficient-Tuning?style=social)
![GitHub Code License](https://img.shields.io/github/license/hiyouga/LLaMA-Efficient-Tuning)
![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Efficient-Tuning)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)

👋 Join our [WeChat](assets/wechat.jpg).

## Changelog

[23/06/03] Now we support quantized training and inference (aka [QLoRA](https://github.com/artidoro/qlora)). Try `--quantization_bit 4/8` argument to work with quantized model. (experimental feature)

[23/05/31] Now we support training the BLOOM & BLOOMZ models in this repo. Try `--model_name_or_path bigscience/bloomz-7b1-mt` argument to use the BLOOMZ model.

## Supported Models

- [LLaMA](https://github.com/facebookresearch/llama) (7B/13B/33B/65B)
- [BLOOM](https://huggingface.co/bigscience/bloom) & [BLOOMZ](https://huggingface.co/bigscience/bloomz) (560M/1.1B/1.7B/3B/7.1B/176B)

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
- For reward model training:
  - [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
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
- protobuf, cpm_kernels and sentencepiece
- jieba, rouge_chinese and nltk (used at evaluation)
- gradio and mdtex2html (used in web_demo.py)

And **powerful GPUs**!

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

### LLaMA Weights Preparation

1. Download the weights of the LLaMA models.
2. Convert them to HF format using the following command.

```bash
python -m transformers.models.llama.convert_llama_weights_to_hf \
    --input_dir path_to_llama_weights --model_size 7B --output_dir path_to_llama_model
```

### (Continually) Pre-Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_pt.py \
    --model_name_or_path path_to_llama_model \
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
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --checkpoint_dir path_to_pt_checkpoint \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --resume_lora_training False \
    --plot_loss \
    --fp16
```

### Reward Model Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_rm.py \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset comparison_gpt4_en \
    --finetuning_type lora \
    --checkpoint_dir path_to_pt_checkpoint \
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
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --checkpoint_dir path_to_pt_checkpoint,path_to_sft_checkpoint \
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

### Evaluation (BLEU and ROUGE_CHINESE)

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
    --model_name_or_path path_to_llama_model \
    --do_eval \
    --dataset alpaca_gpt4_en \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_eval_result \
    --per_device_eval_batch_size 8 \
    --max_samples 50 \
    --predict_with_generate
```

We recommend using `--per_device_eval_batch_size=1` and `--max_target_length 128` in INT8 evaluation.

### CLI Demo

```bash
python src/cli_demo.py \
    --model_name_or_path path_to_llama_model \
    --checkpoint_dir path_to_checkpoint
```

### Web Demo
```bash
python src/web_demo.py \
    --model_name_or_path path_to_llama_model \
    --checkpoint_dir path_to_checkpoint
```

### Export model

```bash
python src/export_model.py \
    --model_name_or_path path_to_llama_model \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_export
```

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

Please follow the [Model Card](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) to use the LLaMA models.

Please follow the [RAIL License](https://huggingface.co/spaces/bigscience/license) to use the BLOOM & BLOOMZ models.

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
