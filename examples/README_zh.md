我们提供了多样化的大模型微调示例脚本。

请确保在 `LLaMA-Factory` 目录下执行下述命令。

## 目录

- [LoRA 微调](#lora-微调)
- [QLoRA 微调](#qlora-微调)
- [全参数微调](#全参数微调)
- [合并 LoRA 适配器与模型量化](#合并-lora-适配器与模型量化)
- [推理 LoRA 模型](#推理-lora-模型)
- [杂项](#杂项)

使用 `CUDA_VISIBLE_DEVICES`（GPU）或 `ASCEND_RT_VISIBLE_DEVICES`（NPU）选择计算设备。

## 示例

### LoRA 微调

#### （增量）预训练

```bash
llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml
```

#### 指令监督微调

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

#### 多模态指令监督微调

```bash
llamafactory-cli train examples/train_lora/llava1_5_lora_sft.yaml
```

#### 奖励模型训练

```bash
llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml
```

#### PPO 训练

```bash
llamafactory-cli train examples/train_lora/llama3_lora_ppo.yaml
```

#### DPO/ORPO/SimPO 训练

```bash
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
```

#### KTO 训练

```bash
llamafactory-cli train examples/train_lora/llama3_lora_kto.yaml
```

#### 预处理数据集

对于大数据集有帮助，在配置中使用 `tokenized_path` 以加载预处理后的数据集。

```bash
llamafactory-cli train examples/train_lora/llama3_preprocess.yaml
```

#### 在 MMLU/CMMLU/C-Eval 上评估

```bash
llamafactory-cli eval examples/train_lora/llama3_lora_eval.yaml
```

#### 批量预测并计算 BLEU 和 ROUGE 分数

```bash
llamafactory-cli train examples/train_lora/llama3_lora_predict.yaml
```

#### 多机指令监督微调

```bash
FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

#### 使用 DeepSpeed ZeRO-3 平均分配显存

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml
```

### QLoRA 微调

#### 基于 4/8 比特 Bitsandbytes/HQQ/EETQ 量化进行指令监督微调（推荐）

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_otfq.yaml
```

#### 基于 4/8 比特 GPTQ 量化进行指令监督微调

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_gptq.yaml
```

#### 基于 4 比特 AWQ 量化进行指令监督微调

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_awq.yaml
```

#### 基于 2 比特 AQLM 量化进行指令监督微调

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_aqlm.yaml
```

### 全参数微调

#### 在单机上进行指令监督微调

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

#### 在多机上进行指令监督微调

```bash
FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

#### 批量预测并计算 BLEU 和 ROUGE 分数

```bash
llamafactory-cli train examples/train_full/llama3_full_predict.yaml
```

### 合并 LoRA 适配器与模型量化

#### 合并 LoRA 适配器

注：请勿使用量化后的模型或 `quantization_bit` 参数来合并 LoRA 适配器。

```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

#### 使用 AutoGPTQ 量化模型

```bash
llamafactory-cli export examples/merge_lora/llama3_gptq.yaml
```

### 推理 LoRA 模型

#### 使用命令行接口

```bash
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```

#### 使用浏览器界面

```bash
llamafactory-cli webchat examples/inference/llama3_lora_sft.yaml
```

#### 启动 OpenAI 风格 API

```bash
llamafactory-cli api examples/inference/llama3_lora_sft.yaml
```

### 杂项

#### 使用 GaLore 进行全参数训练

```bash
llamafactory-cli train examples/extras/galore/llama3_full_sft.yaml
```

#### 使用 BAdam 进行全参数训练

```bash
llamafactory-cli train examples/extras/badam/llama3_full_sft.yaml
```

#### LoRA+ 微调

```bash
llamafactory-cli train examples/extras/loraplus/llama3_lora_sft.yaml
```

#### PiSSA 微调

```bash
llamafactory-cli train examples/extras/pissa/llama3_lora_sft.yaml
```

#### 深度混合微调

```bash
llamafactory-cli train examples/extras/mod/llama3_full_sft.yaml
```

#### LLaMA-Pro 微调

```bash
bash examples/extras/llama_pro/expand.sh
llamafactory-cli train examples/extras/llama_pro/llama3_freeze_sft.yaml
```

#### FSDP+QLoRA 微调

```bash
bash examples/extras/fsdp_qlora/train.sh
```
