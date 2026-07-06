# 快速开始

LLaMA Factory 是一个高效、灵活的大模型微调框架，支持 100+ 种主流大语言模型的微调训练。本文档将帮助您快速上手使用 LLaMA Factory。

## 支持的训练方法

|          方法          |     全参数训练      |    部分参数训练     |       LoRA         |       QLoRA        |
|:---------------------:| ------------------ | ------------------ | ------------------ | ------------------ |
|   指令监督微调 (SFT)   | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
|      DPO 训练          | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

> **提示**: v1 版本目前支持 SFT 和 DPO 两种训练方法，均支持多种加速特性，包括 DeepSpeed、FSDP、FlashAttention-2 等。

## 软件依赖

|          必需项          | 至少     | 推荐     |
|:---------------------:|--------|--------|
|        python         | 3.11   | 3.12   |
|         torch         | 2.7.1  | 2.7.1  |
| torch-npu(Ascend NPU) | 2.7.1  | 2.7.1  |
|      torchvision      | 0.22.1 | 0.22.1 |
|     transformers      | 5.0.0  | 5.0.0  |
|       datasets        | 3.2.0  | 4.0.0  |
|         peft          | 0.18.1 | 0.18.1 |


|       可选项        | 至少     | 推荐     |
|:----------------:|--------|--------|
| CUDA(NVIDIA GPU) | 11.6   | 12.2   |
|    deepspeed     | 0.18.4 | 0.18.4 |
|   flash-attn(NVIDIA GPU)   | 2.5.6  | 2.7.2  |


## 安装 LLaMA Factory

> [!IMPORTANT]
> 此步骤为必需。请确保您的环境满足上述软件依赖要求。

### 从源码安装（推荐）

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

### 使用 pip 安装

```bash
pip install llamafactory
```

### 可选依赖

如果您需要使用特定的加速特性，可以安装相应的依赖：

```bash
# 安装 FlashAttention-2 支持
pip install flash-attn --no-build-isolation

# 安装 DeepSpeed 支持
pip install deepspeed

# 安装 Unsloth 支持（用于加速 LoRA 训练）
pip install unsloth
```

## 数据准备

LLaMA Factory 支持多种数据格式，包括 JSON、JSONL、CSV 等。关于数据集文件的详细格式说明，请参考 [数据准备指南](../../data/README_zh.md)。

### 使用内置数据集

LLaMA Factory 提供了多个内置数据集用于快速测试，您可以在 `data/dataset_info.json` 中查看所有可用的数据集。

### 使用自定义数据集

您可以使用 HuggingFace / ModelScope 上的数据集或加载本地数据集。

> [!NOTE]
> 使用自定义数据集或自定义数据集格式时，请参照 [数据准备指南](../../data/README_zh.md) 进行配置。如有必要，请重新实现自定义数据集的数据处理逻辑，包括对应的 `converter`。

### 数据构建工具

您也可以使用以下工具构建用于微调的合成数据：
- **[Easy Dataset](https://github.com/ConardLi/easy-dataset)** - 易于使用的数据集构建工具
- **[DataFlow](https://github.com/OpenDCAI/DataFlow)** - 高质量数据准备管道
- **[GraphGen](https://github.com/open-sciencelab/GraphGen)** - 基于图的数据生成工具

## 快速开始

### 命令行训练

下面的命令展示了对 Qwen3-0.6B 模型使用 FSDP2 进行全参数微调：

```bash
export USE_V1=1
llamafactory-cli sft examples/v1/train_full/train_full_fsdp2.yaml
```

> **提示**: `llamafactory-cli sft` 和 `llamafactory-cli train` 命令等价。

### Web UI 训练

LLaMA Factory 提供了直观的 Web 界面（LLaMA Board），您可以通过图形界面进行训练：

```bash
llamafactory-cli webui
```

在浏览器中打开 http://localhost:7860 即可开始使用。

### 推理部署

训练完成后，您可以使用以下命令部署模型：

```bash
# 使用 vLLM 后端进行高性能推理
llamafactory-cli chat --model_name_or_path path/to/your/model --template qwen --infer_backend vllm

# 使用 HuggingFace 后端进行推理
llamafactory-cli chat --model_name_or_path path/to/your/model --template qwen
```

## 进阶用法

高级用法请参考 [进阶指南](./advanced/README.md)，包括：
- 多卡多机分布式训练
- LoRA/QLoRA 微调
- 模型量化（AWQ/GPTQ/LLM.int8 等）
- 各种加速特性（DeepSpeed、FSDP、FlashAttention 等）
- 多模态模型微调

## 常见问题

### 1. 内存不足怎么办？

- 使用 LoRA 或 QLoRA 代替全参数训练
- 减小 `batch_size` 和 `cutoff_len`
- 启用 `gradient_checkpointing`
- 使用 DeepSpeed ZeRO-2 或 ZeRO-3

### 2. 如何选择合适的训练方法？

- **SFT（指令微调）**: 最常用的方法，适用于大多数场景，通过监督数据训练模型
- **DPO（直接偏好优化）**: 用于对齐人类偏好，提升模型输出质量，无需训练奖励模型

### 3. 训练完成后如何评估模型？

```bash
llamafactory-cli eval --model_name_or_path path/to/your/model --template qwen --dataset mmlu
```

更多评估方法请参考 [评估指南](./advanced/evaluation.md)。

## 获取帮助

如果您在使用过程中遇到问题：
- 查看 [GitHub Issues](https://github.com/hiyouga/LLaMA-Factory/issues)
- 加入 [Discord 社区](https://discord.gg/rKfvV9r9FK)
- 加入微信群（扫描二维码）
