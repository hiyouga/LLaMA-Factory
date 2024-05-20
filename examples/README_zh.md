我们提供了多样化的大模型微调示例脚本。

请确保在 `LLaMA-Factory` 目录下执行下述命令。

## 目录

- [单 GPU LoRA 微调](#单-gpu-lora-微调)
- [单 GPU QLoRA 微调](#单-gpu-qlora-微调)
- [多 GPU LoRA 微调](#多-gpu-lora-微调)
- [多 NPU LoRA 微调](#多-npu-lora-微调)
- [多 GPU 全参数微调](#多-gpu-全参数微调)
- [合并 LoRA 适配器与模型量化](#合并-lora-适配器与模型量化)
- [推理 LoRA 模型](#推理-lora-模型)
- [杂项](#杂项)

## 示例

### 单 GPU LoRA 微调

#### （增量）预训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_pretrain.yaml
```

#### 指令监督微调

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_sft.yaml
```

#### 多模态指令监督微调

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llava1_5_lora_sft.yaml
```

#### 奖励模型训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_reward.yaml
```

#### PPO 训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_ppo.yaml
```

#### DPO 训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_dpo.yaml
```

#### KTO 训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_kto.yaml
```

#### ORPO 训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_orpo.yaml
```

#### 预处理数据集

对于大数据集有帮助，在配置中使用 `tokenized_path` 以加载预处理后的数据集。

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_preprocess.yaml
```

#### 在 MMLU/CMMLU/C-Eval 上评估

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli eval examples/lora_single_gpu/llama3_lora_eval.yaml
```

#### 批量预测并计算 BLEU 和 ROUGE 分数

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_predict.yaml
```

### 单 GPU QLoRA 微调

#### 基于 4/8 比特 Bitsandbytes 量化进行指令监督微调（推荐）

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/qlora_single_gpu/llama3_lora_sft_bitsandbytes.yaml
```

#### 基于 4/8 比特 GPTQ 量化进行指令监督微调

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/qlora_single_gpu/llama3_lora_sft_gptq.yaml
```

#### 基于 4 比特 AWQ 量化进行指令监督微调

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/qlora_single_gpu/llama3_lora_sft_awq.yaml
```

#### 基于 2 比特 AQLM 量化进行指令监督微调

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/qlora_single_gpu/llama3_lora_sft_aqlm.yaml
```

### 多 GPU LoRA 微调

#### 使用 Accelerate 进行单节点训练

```bash
bash examples/lora_multi_gpu/single_node.sh
```

#### 使用 Accelerate 进行多节点训练

```bash
bash examples/lora_multi_gpu/multi_node.sh
```

#### 使用 DeepSpeed ZeRO-3 平均分配显存

```bash
bash examples/lora_multi_gpu/ds_zero3.sh
```

### 多 NPU LoRA 微调

#### 使用 DeepSpeed ZeRO-0 训练

```bash
bash examples/lora_multi_npu/ds_zero0.sh
```

### 多 GPU 全参数微调

#### 使用 DeepSpeed 进行单节点训练

```bash
bash examples/full_multi_gpu/single_node.sh
```

#### 使用 DeepSpeed 进行多节点训练

```bash
bash examples/full_multi_gpu/multi_node.sh
```

#### 批量预测并计算 BLEU 和 ROUGE 分数

```bash
bash examples/full_multi_gpu/predict.sh
```

### 合并 LoRA 适配器与模型量化

#### 合并 LoRA 适配器

注：请勿使用量化后的模型或 `quantization_bit` 参数来合并 LoRA 适配器。

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

#### 使用 AutoGPTQ 量化模型

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export examples/merge_lora/llama3_gptq.yaml
```

### 推理 LoRA 模型

使用 `CUDA_VISIBLE_DEVICES=0,1` 进行多卡推理。

#### 使用命令行接口

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```

#### 使用浏览器界面

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat examples/inference/llama3_lora_sft.yaml
```

#### 启动 OpenAI 风格 API

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli api examples/inference/llama3_lora_sft.yaml
```

### 杂项

#### 使用 GaLore 进行全参数训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/galore/llama3_full_sft.yaml
```

#### 使用 BAdam 进行全参数训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/badam/llama3_full_sft.yaml
```

#### LoRA+ 微调

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/loraplus/llama3_lora_sft.yaml
```

#### 深度混合微调

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/mod/llama3_full_sft.yaml
```

#### LLaMA-Pro 微调

```bash
bash examples/extras/llama_pro/expand.sh
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/llama_pro/llama3_freeze_sft.yaml
```

#### FSDP+QLoRA 微调

```bash
bash examples/extras/fsdp_qlora/single_node.sh
```
