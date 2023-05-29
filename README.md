# LLaMA Efficient Tuning

![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/ChatGLM-Efficient-Tuning?style=social)
![GitHub Code License](https://img.shields.io/github/license/hiyouga/ChatGLM-Efficient-Tuning)
![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/ChatGLM-Efficient-Tuning)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)

## Requirement

- Python 3.8+ and PyTorch 1.13.1
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
2. Convert them to HF format using this [script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)

```python
python convert_llama_weights_to_hf.py \
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

This repository is licensed under the [Apache-2.0 License](LICENSE). Please follow the [Model Card](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) to use the LLaMA model.

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
