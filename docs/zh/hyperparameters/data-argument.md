# 数据参数（DataArguments）

数据参数用于配置训练和评估所使用的数据集。

## 基础参数

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `train_dataset` | `str` | `None` | 训练数据集的路径，指向一个数据集配置 YAML 文件（详见下方[数据集配置文件格式](#数据集配置文件格式)）。 |
| `eval_dataset` | `str` | `None` | 评估数据集的路径，格式与 `train_dataset` 相同。 |

### 配置示例

```yaml
train_dataset: data/v1_sft_demo.yaml
eval_dataset: data/v1_eval_demo.yaml
```

## 数据集配置文件格式

`train_dataset` 和 `eval_dataset` 指向的是一个 YAML 文件，该文件中可以定义一个或多个数据集。每个数据集作为一个顶层 key，其下包含数据集的详细配置。

### 数据集配置字段

| 字段名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `path` | `str` | **必填** | 数据集路径。可以是本地文件路径（如 `data/identity.json`）、本地目录路径、或 HuggingFace Hub 上的数据集标识符。 |
| `source` | `str` | `"hf_hub"` | 数据集来源。可选值：`hf_hub`（HuggingFace Hub）、`ms_hub`（ModelScope Hub）、`local`（本地文件）。 |
| `split` | `str` | `"train"` | 数据集分割名称，如 `train`、`test`、`validation` 等。 |
| `converter` | `str` | `None` | 数据格式转换器名称。用于将原始数据集格式转换为框架内部格式。内置转换器包括 `alpaca`、`sharegpt`、`pair`（详见下方[数据格式转换器](#数据格式转换器)）。 |
| `size` | `int` | 全部样本 | 指定从数据集中采样的样本数量。如果设置，将从数据集中随机采样指定数量的样本。 |
| `weight` | `float` | `1.0` | 数据集采样权重。用于多数据集混合训练时调整各数据集的采样比例。例如，若数据集 A 的权重为 1.0，数据集 B 的权重为 0.5，则在混合采样时，从 A 中采样的概率将是从 B 中采样的两倍。 |
| `streaming` | `bool` | `false` | 是否使用流式加载。对于本地文件，启用流式加载可减少内存占用。 |

### 数据集配置示例

```yaml
# 单数据集
identity:
  path: data/identity.json
  source: local
  converter: alpaca

# 多数据集混合
identity:
  path: data/identity.json
  source: local
  converter: alpaca
alpaca_en_demo:
  path: data/alpaca_en_demo.json
  source: local
  converter: alpaca
  size: 500
```

## 数据格式转换器

转换器（Converter）负责将不同格式的原始数据转换为框架内部统一的对话格式。

### `alpaca` 转换器

适用于 Alpaca 格式的数据集。原始数据格式如下：

```json
{
  "system": "系统提示（可选）",
  "instruction": "指令文本",
  "input": "输入文本（可选）",
  "output": "输出文本"
}
```

参考数据集：[llamafactory/alpaca_gpt4_en](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_en)

### `sharegpt` 转换器

适用于 ShareGPT 格式的多轮对话数据集。原始数据格式如下：

```json
{
  "conversations": [
    {"from": "system", "value": "系统提示"},
    {"from": "human", "value": "用户消息"},
    {"from": "gpt", "value": "助手回复"},
    {"from": "function_call", "value": "{\"name\": \"func\", \"arguments\": {}}"},
    {"from": "observation", "value": "工具返回结果"}
  ],
  "tools": "[{\"type\": \"function\", ...}]"
}
```

支持的角色标签：`system`、`human`、`gpt`、`function_call`、`observation`。

参考数据集：[llamafactory/glaive_toolcall_en](https://huggingface.co/datasets/llamafactory/glaive_toolcall_en)

### `pair` 转换器

适用于偏好对（Preference Pair）格式的数据集，用于 DPO 等偏好训练。原始数据格式如下：

```json
{
  "chosen": [
    {"role": "user", "content": "用户消息"},
    {"role": "assistant", "content": "优选回复"}
  ],
  "rejected": [
    {"role": "user", "content": "用户消息"},
    {"role": "assistant", "content": "劣选回复"}
  ]
}
```

参考数据集：[HuggingFaceH4/orca_dpo_pairs](https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs)

> [!NOTE]
> 如需使用自定义数据格式，可以通过实现自定义的 `DataConverterPlugin` 来扩展转换器。详见 [data-preparation](../data-preparation/README.md)。
