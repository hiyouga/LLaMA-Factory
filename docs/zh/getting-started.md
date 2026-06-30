# Getting Started


## 训练方法

|          方法          |     全参数训练      |    部分参数训练     |       LoRA         |       QLoRA        |
|:---------------------:| ------------------ | ------------------ | ------------------ | ------------------ |
|      指令监督微调       | ✅ |  |  | |
|      奖励模型训练       |  |  |  | |
|        DPO 训练        |  |  |  | |




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


## 如何使用

### 安装 LLaMA Factory

> [!IMPORTANT]
> 此步骤为必需。

#### 从源码安装

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
```


### 数据准备

关于数据集文件的格式，请参考 [data-preparation/README.md](data-preparation/README.md) 的内容。你可以使用 HuggingFace / ModelScope 上的数据集或加载本地数据集。

> [!NOTE]
> 使用自定义数据集或自定义数据集格式时，请参照 [data-preparation/README.md](data-preparation/README.md) 进行配置，如有必要，请重新实现自定义数据集的数据处理逻辑，包括对应的`converter`。

您也可以使用 **[Easy Dataset](https://github.com/ConardLi/easy-dataset)**、**[DataFlow](https://github.com/OpenDCAI/DataFlow)** 和 **[GraphGen](https://github.com/open-sciencelab/GraphGen)** 构建用于微调的合成数据。

### 快速开始

下面的命令展示了对 Qwen3-0.6B 模型使用 FSDP2 进行 全参**微调**，两行命令等价。

```bash
export USE_V1=1
llamafactory-cli sft examples/v1/train_full/train_full_fsdp2.yaml
llamafactory-cli train examples/v1/train_full/train_full_fsdp2.yaml

```

高级用法请参考 [advanced](./advanced/README.md)（包括多卡多机微调、分布式、Lora、量化、以及各种加速特性等）。
