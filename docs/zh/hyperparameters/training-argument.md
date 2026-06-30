# 训练参数（TrainingArguments）

训练参数用于控制训练过程的核心超参数，包括批量大小、学习率、训练轮数等，以及分布式训练、优化器和学习率调度器的插件配置。

## 基础参数

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `output_dir` | `str` | `outputs/<random_hex>` | 模型输出目录，用于保存训练好的模型和检查点。若不指定，默认生成一个随机目录名。 |
| `micro_batch_size` | `int` | `1` | 每个设备上的微批量大小（即每次前向/反向传播处理的样本数）。 |
| `global_batch_size` | `int` | `None` | 全局批量大小。默认值为 `数据并行数 × micro_batch_size`。当显式设置时，框架将自动计算梯度累积步数（`global_batch_size / (dp_size * micro_batch_size)`）。 |
| `cutoff_len` | `int` | `2048` | 最大序列长度。超过此长度的序列将被截断。 |
| `learning_rate` | `float` | `1e-4` | 学习率。 |
| `num_train_epochs` | `int` | `3` | 训练总轮数。 |
| `max_steps` | `int` | `None` | 最大训练步数。若设置，将覆盖 `num_train_epochs`，以步数为准进行训练。 |
| `max_grad_norm` | `float` | `1.0` | 梯度裁剪的最大范数。用于防止梯度爆炸。 |
| `bf16` | `bool` | `false` | 是否使用 bfloat16 精度进行训练。建议在支持 bf16 的硬件（如 A100、H100）上开启以加速训练并降低显存占用。 |
| `enable_activation_checkpointing` | `bool` | `true` | 是否启用激活检查点（Activation Checkpointing / Gradient Checkpointing）。通过以计算换内存的方式，显著降低训练时的显存占用。默认开启。 |
| `batching_strategy` | `str` | `"normal"` | 批处理策略。可选值见下方 [批处理策略](#批处理策略)。 |
| `batching_workers` | `int` | `16` | 批处理数据加载的工作线程数。 |

### 配置示例

```yaml
output_dir: outputs/my_model
micro_batch_size: 2
global_batch_size: 16
cutoff_len: 4096
learning_rate: 2.0e-5
num_train_epochs: 3
max_grad_norm: 1.0
bf16: true
enable_activation_checkpointing: true
batching_strategy: normal
```

## 批处理策略

`batching_strategy` 支持以下策略：

| 策略值 | 说明 |
|:---:|:---|
| `normal` | 标准批处理，所有序列填充到相同长度。 |
| `padding_free` | 无填充批处理，将多条样本拼接为一个长序列以消除 padding 开销，提高训练效率。 |
| `dynamic_batching` | 动态批处理，根据序列实际长度动态组装批次，减少不必要的 padding。 |
| `dynamic_padding_free` | 动态无填充批处理，结合动态批处理和无填充的优势。 |

## 插件配置（PluginConfig）

训练参数中的 `dist_config`、`optim_config`、`lr_scheduler_config` 均为插件配置类型。插件配置是一个字典，必须包含 `name` 字段来指定使用哪个插件，其余字段为该插件的特定参数。

### 分布式训练配置（dist_config）

用于配置分布式训练策略。

#### FSDP2

PyTorch 原生的 Fully Sharded Data Parallel (FSDP2) 分布式训练。

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `name` | `str` | **必填** | 固定为 `"fsdp2"`。 |
| `mixed_precision` | `str` | `"bf16"` | 混合精度策略。可选值：`"bf16"`、`"fp16"`、`"fp32"`。 |
| `reshard_after_forward` | `bool` | `true` | 前向传播后是否重新分片参数。设为 `true` 可降低显存占用，但增加通信开销。 |
| `offload_params` | `bool` | `false` | 是否将参数卸载到 CPU。可显著降低 GPU 显存占用，但会降低训练速度。 |
| `pin_memory` | `bool` | `true` | CPU 卸载时是否使用 pinned memory。与 `offload_params` 配合使用。 |
| `dcp_path` | `str` | `None` | 分布式检查点（DCP）路径。若指定，将优先从 DCP 加载模型权重，加载速度更快。 |

**配置示例：**

```yaml
dist_config:
  name: fsdp2
  mixed_precision: bf16
  reshard_after_forward: true
  offload_params: false
  dcp_path: null
```

#### DeepSpeed

通过 Accelerate 集成的 DeepSpeed 分布式训练。

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `name` | `str` | **必填** | 固定为 `"deepspeed"`。 |
| `config_file` | `str` | **必填** | DeepSpeed 配置文件路径（JSON 格式）。该文件包含 ZeRO 优化、混合精度、通信等所有 DeepSpeed 相关配置。 |

**配置示例：**

```yaml
dist_config:
  name: deepspeed
  config_file: examples/deepspeed/ds_z3_config.json
```

### 优化器配置（optim_config）

用于配置自定义优化器插件。默认值为 `None`，即使用框架内置的 AdamW 优化器。

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `name` | `str` | **必填** | 优化器插件名称。 |

> [!NOTE]
> 优化器插件通过 `OptimizerPlugin` 扩展。可通过自定义插件实现对其他优化器（如 LION、Sophia 等）的支持。

### 学习率调度器配置（lr_scheduler_config）

用于配置自定义学习率调度器插件。默认值为 `None`，即使用框架内置的学习率调度策略。

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `name` | `str` | **必填** | 学习率调度器插件名称。 |

> [!NOTE]
> 学习率调度器插件通过 `LRSchedulerPlugin` 扩展。可通过自定义插件实现对其他调度策略的支持。
