# 插件配置

v1 通过 YAML 中的 `*_config` 字段选择模型和训练插件。每个配置都是一个
映射，其中 `name` 指定实现，其余字段由该实现解析。

```yaml
peft_config:
  name: lora
  r: 16
  target_modules: all

dist_config:
  name: fsdp2
  reshard_after_forward: true

optim_config:
  name: muon
  wd: 0.1
```

同一插件配置中不能混用其他实现的专属字段。未声明的字段会触发参数解析
错误。

## 模型插件

| YAML 字段 | 可用名称 | 用途 | 参数 |
|-----------|----------|------|------|
| `init_config` | `init_on_default`、`init_on_meta`、`init_on_rank0` | 选择模型初始化设备 | [模型参数](model.md#init_config) |
| `peft_config` | `lora`、`freeze` | 配置参数高效微调或冻结训练 | [模型参数](model.md#peft_config) |
| `quant_config` | `auto`、`bnb` | 配置 bitsandbytes 量化 | [模型参数](model.md#quant_config) |
| `kernel_config` | `auto`、Kernel 实现名称 | 应用一个或多个融合算子加速实现 | [模型参数](model.md#kernel_config) |

`kernel_config.name` 可以使用逗号分隔多个实现，应用顺序与配置顺序一致：

```yaml
kernel_config:
  name: npu_fused_rmsnorm,npu_fused_rope
```

## 训练插件

| YAML 字段 | 可用名称 | 用途 | 参数 |
|-----------|----------|------|------|
| `dist_config` | `fsdp2`、`deepspeed` | 选择分布式训练后端 | [训练参数](training.md#dist_config) |
| `optim_config` | `muon` | 使用 Muon 优化器 | [训练参数](training.md#optim_config) |

`learning_rate` 始终使用顶层训练参数。配置 Muon 时不需要在
`optim_config` 中重复设置学习率。

## 其他实现选择

以下功能同样通过名称选择实现，但不使用 `*_config` 映射：

| YAML 字段 | 可用名称 | 用途 |
|-----------|----------|------|
| `converter` | `alpaca`、`sharegpt`、`pair` | 将数据集转换为 v1 Messages 格式 |
| `batching_strategy` | `normal`、`padding_free`、`dynamic_batching`、`dynamic_padding_free` | 选择批处理策略 |
| `cp_mode` | `ulysses` | 选择 Context Parallel 实现 |
| `sample_backend` | `hf` | 选择推理采样后端 |

数据转换、批处理和分布式训练的完整用法分别见[数据准备](../feature-guide/data_preparation.md)、
[批处理策略](../feature-guide/batching.md)和[分布式训练](../feature-guide/distributed_training.md)。
