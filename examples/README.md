We provide diverse examples about fine-tuning LLMs.

Make sure to execute these commands in the `LLaMA-Factory` directory.

## Table of Contents

- [LoRA Fine-Tuning](#lora-fine-tuning)
- [QLoRA Fine-Tuning](#qlora-fine-tuning)
- [Full-Parameter Fine-Tuning](#full-parameter-fine-tuning)
- [Merging LoRA Adapters and Quantization](#merging-lora-adapters-and-quantization)
- [Inferring LoRA Fine-Tuned Models](#inferring-lora-fine-tuned-models)
- [Extras](#extras)

Use `CUDA_VISIBLE_DEVICES` (GPU) or `ASCEND_RT_VISIBLE_DEVICES` (NPU) to choose computing devices.

## Examples

### LoRA Fine-Tuning

#### (Continuous) Pre-Training

```bash
llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml
```

#### Supervised Fine-Tuning

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

#### Multimodal Supervised Fine-Tuning

```bash
llamafactory-cli train examples/train_lora/llava1_5_lora_sft.yaml
```

#### Reward Modeling

```bash
llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml
```

#### PPO Training

```bash
llamafactory-cli train examples/train_lora/llama3_lora_ppo.yaml
```

#### DPO/ORPO/SimPO Training

```bash
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
```

#### KTO Training

```bash
llamafactory-cli train examples/train_lora/llama3_lora_kto.yaml
```

#### Preprocess Dataset

It is useful for large dataset, use `tokenized_path` in config to load the preprocessed dataset.

```bash
llamafactory-cli train examples/train_lora/llama3_preprocess.yaml
```

#### Evaluating on MMLU/CMMLU/C-Eval Benchmarks

```bash
llamafactory-cli eval examples/train_lora/llama3_lora_eval.yaml
```

#### Batch Predicting and Computing BLEU and ROUGE Scores

```bash
llamafactory-cli train examples/train_lora/llama3_lora_predict.yaml
```

#### Supervised Fine-Tuning on Multiple Nodes

```bash
FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

#### Supervised Fine-Tuning with DeepSpeed ZeRO-3 (Weight Sharding)

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml
```

### QLoRA Fine-Tuning

#### Supervised Fine-Tuning with 4/8-bit Bitsandbytes/HQQ/EETQ Quantization (Recommended)

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_otfq.yaml
```

#### Supervised Fine-Tuning with 4/8-bit GPTQ Quantization

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_gptq.yaml
```

#### Supervised Fine-Tuning with 4-bit AWQ Quantization

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_awq.yaml
```

#### Supervised Fine-Tuning with 2-bit AQLM Quantization

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_aqlm.yaml
```

### Full-Parameter Fine-Tuning

#### Supervised Fine-Tuning on Single Node

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

#### Supervised Fine-Tuning on Multiple Nodes

```bash
FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

#### Batch Predicting and Computing BLEU and ROUGE Scores

```bash
llamafactory-cli train examples/train_full/llama3_full_predict.yaml
```

### Merging LoRA Adapters and Quantization

#### Merge LoRA Adapters

Note: DO NOT use quantized model or `quantization_bit` when merging LoRA adapters.

```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

#### Quantizing Model using AutoGPTQ

```bash
llamafactory-cli export examples/merge_lora/llama3_gptq.yaml
```

### Inferring LoRA Fine-Tuned Models

#### Use CLI

```bash
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```

#### Use Web UI

```bash
llamafactory-cli webchat examples/inference/llama3_lora_sft.yaml
```

#### Launch OpenAI-style API

```bash
llamafactory-cli api examples/inference/llama3_lora_sft.yaml
```

### Extras

#### Full-Parameter Fine-Tuning using GaLore

```bash
llamafactory-cli train examples/extras/galore/llama3_full_sft.yaml
```

#### Full-Parameter Fine-Tuning using BAdam

```bash
llamafactory-cli train examples/extras/badam/llama3_full_sft.yaml
```

#### Full-Parameter Fine-Tuning using Adam-mini

```bash
llamafactory-cli train examples/extras/adam_mini/qwen2_full_sft.yaml
```

#### LoRA+ Fine-Tuning

```bash
llamafactory-cli train examples/extras/loraplus/llama3_lora_sft.yaml
```

#### PiSSA Fine-Tuning

```bash
llamafactory-cli train examples/extras/pissa/llama3_lora_sft.yaml
```

#### Mixture-of-Depths Fine-Tuning

```bash
llamafactory-cli train examples/extras/mod/llama3_full_sft.yaml
```

#### LLaMA-Pro Fine-Tuning

```bash
bash examples/extras/llama_pro/expand.sh
llamafactory-cli train examples/extras/llama_pro/llama3_freeze_sft.yaml
```

#### FSDP+QLoRA Fine-Tuning

```bash
bash examples/extras/fsdp_qlora/train.sh
```
