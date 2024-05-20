We provide diverse examples about fine-tuning LLMs.

Make sure to execute these commands in the `LLaMA-Factory` directory.

## Table of Contents

- [LoRA Fine-Tuning on A Single GPU](#lora-fine-tuning-on-a-single-gpu)
- [QLoRA Fine-Tuning on a Single GPU](#qlora-fine-tuning-on-a-single-gpu)
- [LoRA Fine-Tuning on Multiple GPUs](#lora-fine-tuning-on-multiple-gpus)
- [LoRA Fine-Tuning on Multiple NPUs](#lora-fine-tuning-on-multiple-npus)
- [Full-Parameter Fine-Tuning on Multiple GPUs](#full-parameter-fine-tuning-on-multiple-gpus)
- [Merging LoRA Adapters and Quantization](#merging-lora-adapters-and-quantization)
- [Inferring LoRA Fine-Tuned Models](#inferring-lora-fine-tuned-models)
- [Extras](#extras)

## Examples

### LoRA Fine-Tuning on A Single GPU

#### (Continuous) Pre-Training

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_pretrain.yaml
```

#### Supervised Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_sft.yaml
```

#### Multimodal Supervised Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llava1_5_lora_sft.yaml
```

#### Reward Modeling

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_reward.yaml
```

#### PPO Training

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_ppo.yaml
```

#### DPO Training

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_dpo.yaml
```

#### KTO Training

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_kto.yaml
```

#### ORPO Training

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_orpo.yaml
```

#### Preprocess Dataset

It is useful for large dataset, use `tokenized_path` in config to load the preprocessed dataset.

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_preprocess.yaml
```

#### Evaluating on MMLU/CMMLU/C-Eval Benchmarks

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli eval examples/lora_single_gpu/llama3_lora_eval.yaml
```

#### Batch Predicting and Computing BLEU and ROUGE Scores

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_predict.yaml
```

### QLoRA Fine-Tuning on a Single GPU

#### Supervised Fine-Tuning with 4/8-bit Bitsandbytes Quantization (Recommended)

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/qlora_single_gpu/llama3_lora_sft_bitsandbytes.yaml
```

#### Supervised Fine-Tuning with 4/8-bit GPTQ Quantization

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/qlora_single_gpu/llama3_lora_sft_gptq.yaml
```

#### Supervised Fine-Tuning with 4-bit AWQ Quantization

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/qlora_single_gpu/llama3_lora_sft_awq.yaml
```

#### Supervised Fine-Tuning with 2-bit AQLM Quantization

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/qlora_single_gpu/llama3_lora_sft_aqlm.yaml
```

### LoRA Fine-Tuning on Multiple GPUs

#### Supervised Fine-Tuning with Accelerate on Single Node

```bash
bash examples/lora_multi_gpu/single_node.sh
```

#### Supervised Fine-Tuning with Accelerate on Multiple Nodes

```bash
bash examples/lora_multi_gpu/multi_node.sh
```

#### Supervised Fine-Tuning with DeepSpeed ZeRO-3 (Weight Sharding)

```bash
bash examples/lora_multi_gpu/ds_zero3.sh
```

### LoRA Fine-Tuning on Multiple NPUs

#### Supervised Fine-Tuning with DeepSpeed ZeRO-0

```bash
bash examples/lora_multi_npu/ds_zero0.sh
```

### Full-Parameter Fine-Tuning on Multiple GPUs

#### Supervised Fine-Tuning with Accelerate on Single Node

```bash
bash examples/full_multi_gpu/single_node.sh
```

#### Supervised Fine-Tuning with Accelerate on Multiple Nodes

```bash
bash examples/full_multi_gpu/multi_node.sh
```

#### Batch Predicting and Computing BLEU and ROUGE Scores

```bash
bash examples/full_multi_gpu/predict.sh
```

### Merging LoRA Adapters and Quantization

#### Merge LoRA Adapters

Note: DO NOT use quantized model or `quantization_bit` when merging LoRA adapters.

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

#### Quantizing Model using AutoGPTQ

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export examples/merge_lora/llama3_gptq.yaml
```

### Inferring LoRA Fine-Tuned Models

Use `CUDA_VISIBLE_DEVICES=0,1` to infer models on multiple devices.

#### Use CLI

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```

#### Use Web UI

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat examples/inference/llama3_lora_sft.yaml
```

#### Launch OpenAI-style API

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli api examples/inference/llama3_lora_sft.yaml
```

### Extras

#### Full-Parameter Fine-Tuning using GaLore

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/galore/llama3_full_sft.yaml
```

#### Full-Parameter Fine-Tuning using BAdam

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/badam/llama3_full_sft.yaml
```

#### LoRA+ Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/loraplus/llama3_lora_sft.yaml
```

#### Mixture-of-Depths Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/mod/llama3_full_sft.yaml
```

#### LLaMA-Pro Fine-Tuning

```bash
bash examples/extras/llama_pro/expand.sh
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/llama_pro/llama3_freeze_sft.yaml
```

#### FSDP+QLoRA Fine-Tuning

```bash
bash examples/extras/fsdp_qlora/single_node.sh
```
