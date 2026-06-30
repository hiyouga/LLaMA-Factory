# SFT

Supervised Fine-Tuning(监督微调)是一种在预训练模型上使用小规模有标签数据集进行训练的方法。 相比于预训练一个全新的模型，对已有的预训练模型进行监督微调是更快速更节省成本的途径。

要启动监督微调，请参考如下配置文件示例进行配置：

```yaml
### example_sft.yaml

model: Qwen/Qwen3-0.6B

model_class: llm

template: qwen3_nothink

kernel_config:
    name: auto
    include_kernels: auto

dist_config:
    name: deepspeed
    config_file: examples/deepspeed/ds_z3_config.json

### data
train_dataset: data/v1_sft_demo.yaml

### training
output_dir: outputs/Qwen3-0.6B-deepspeed
micro_batch_size: 1
cutoff_len: 2048
learning_rate: 1.0e-4
bf16: true
max_steps: 10

```
并通过如下命令启动 sft 微调：

```bash
export USE_V1=1

llamafactory-cli sft example_sft.yaml

# 或者使用如下命令，二者等价
llamafactory-cli train example_sft.yaml

```

> [!NOTE]
> 在 v0 版本中，`llamafactory-cli train` 命令可用于启动所有训练，包括预训练、sft、dpo 等，具体行为通过配置文件中的 **stage** 字段参数决定；但是在 v1 中，**stage** 字段已废弃，`llamafactory-cli train` 等价于 `llamafactory-cli sft`，仅用于启动 sft 训练。若要进行 dpo，rm 等 stage 训练，可使用 `llamafactory-cli dpo`，`llamafactory-cli rm` 等命令来启动，具体参见[dpo](./dpo.md) 、 [rm](./rm.md)。
