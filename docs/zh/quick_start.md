# 快速开始

> v1 是 LlamaFactory 的实验性架构，通过 `USE_V1=1` 启用。稳定版（v0）的
> 功能与完整文档见 [LlamaFactory 文档](https://llamafactory.readthedocs.io/)。

本页是 v1 的快速指南，介绍安装、启用 v1 和运行 SFT。更完整的环境说明
见仓库 README。

## 安装

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
```

## 启用 v1

```bash
export USE_V1=1
```

## 运行 SFT

仓库提供数据配置 `data/v1_sft_demo.yaml` 和可运行示例：

```bash
llamafactory-cli sft examples/v1/train_full/train_full_fsdp2.yaml
```

检测到多个设备时，CLI 会自动通过 `torchrun` 重启；单设备需要强制走
该入口时可以设置 `FORCE_TORCHRUN=1`。

## 选择运行命令

v1 当前支持以下命令：

| 命令 | 用途 |
|------|------|
| `sft` | 监督微调 |
| `dpo` | 偏好优化 |
| `rm` | 奖励模型训练 |
| `chat` | 交互式推理 |
| `merge` | 合并 LoRA adapter 并导出模型 |

## 其他

- 参数字段和插件配置：[参数说明](configuration/index.md)
- 数据格式与数据集混合：[数据准备](feature-guide/data_preparation.md)
- 全参、LoRA、Freeze 与 QLoRA：[SFT](feature-guide/sft.md)
- DPO 与 RM：[DPO](feature-guide/dpo.md) / [RM](feature-guide/rm.md)
- 批处理策略：[批处理](feature-guide/batching.md)
- FSDP2、DeepSpeed 与 Ulysses：[分布式训练](feature-guide/distributed_training.md)
- v1 内部结构：[开发者指南](developer-guide/index.md)
