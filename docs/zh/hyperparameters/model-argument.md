# 模型参数（ModelArguments）

模型参数用于配置模型加载、模板选择、PEFT（参数高效微调）、量化以及算子优化等。

## 基础参数

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `model` | `str` | `"Qwen/Qwen3-4B-Instruct-2507"` | 模型路径或 HuggingFace Hub 上的模型标识符。支持本地路径（如 `/path/to/model`）和远程仓库 ID（如 `Qwen/Qwen3-0.6B`）。 |
| `template` | `str` | `"qwen3_nothink"` | 对话模板名称。用于控制如何将多轮对话渲染为模型输入格式。不同模型需使用对应的模板。 |
| `trust_remote_code` | `bool` | `false` | 是否信任并执行 HuggingFace Hub 上模型仓库中的自定义代码。部分模型（如 ChatGLM）需要设为 `true`。 |
| `model_class` | `str` | `"llm"` | 模型类别。可选值：`"llm"`（因果语言模型）、`"cls"`（分类模型）、`"other"`（其他类型）。 |

### 配置示例

```yaml
model: Qwen/Qwen3-0.6B
template: qwen3_nothink
trust_remote_code: true
model_class: llm
```

## 插件配置

模型参数中的 `init_config`、`peft_config`、`kernel_config`、`quant_config` 均为插件配置类型。

### 模型初始化配置（init_config）

控制模型参数在分布式训练中的初始化位置。

| 插件名 | 说明 |
|:---:|:---|
| `init_on_meta` | 在 meta 设备上初始化模型（不分配实际内存），配合 FSDP2 使用可实现大模型的高效加载。 |
| `init_on_rank0` | 仅在 rank 0 上使用 CPU 初始化模型，其他 rank 使用 meta 设备。 |
| `init_on_default` | 在当前设备（GPU）上直接初始化模型。 |

**配置示例：**

```yaml
init_config:
  name: init_on_meta
```

### PEFT 配置（peft_config）

用于配置参数高效微调方法。目前支持 LoRA 和 Freeze 两种方式。

#### LoRA

Low-Rank Adaptation，通过在目标模块中注入低秩矩阵来实现高效微调。

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `name` | `str` | **必填** | 固定为 `"lora"`。 |
| `r` | `int` | `8` | LoRA 秩（rank）。值越大，可训练参数越多，表达能力越强，但也需要更多显存。常用值：`8`、`16`、`32`、`64`。 |
| `lora_alpha` | `int` | `16` | LoRA 缩放系数。通常设置为 `r` 的 1~2 倍。实际缩放比例为 `lora_alpha / r`。 |
| `lora_dropout` | `float` | `0.05` | LoRA 层的 Dropout 比率。用于防止过拟合。 |
| `target_modules` | `str` 或 `list[str]` | `"all"` | 应用 LoRA 的目标模块名称。设为 `"all"` 将自动查找所有线性层（排除 `lm_head` 等输出层）。也可以指定具体模块名，如 `["q_proj", "v_proj"]`。 |
| `use_rslora` | `bool` | `false` | 是否使用 RS-LoRA（Rank-Stabilized LoRA）。RS-LoRA 使用 `lora_alpha / sqrt(r)` 作为缩放因子，在高秩下更稳定。 |
| `use_dora` | `bool` | `false` | 是否使用 DoRA（Weight-Decomposed Low-Rank Adaptation）。DoRA 将权重分解为方向和大小，通常能获得更好的效果。 |
| `modules_to_save` | `list[str]` | `None` | 额外需要完整保存的模块名称列表。这些模块将同时保存原始权重，通常用于 embedding 层或 lm_head。 |
| `adapter_name_or_path` | `str` 或 `list[str]` | `None` | 已有 LoRA 适配器的路径。训练模式下仅支持传入单个路径（用于断点续训）；推理模式下支持传入多个路径（将自动合并）。 |

**导出相关参数**（用于 `llamafactory-cli export` 命令）：

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `export_dir` | `str` | `None` | 合并导出模型的目标目录。 |
| `export_size` | `int` | `5` | 导出模型的分片大小（单位：GB）。 |
| `export_hub_model_id` | `str` | `None` | 导出后推送到 HuggingFace Hub 的模型 ID。 |
| `infer_dtype` | `str` | `"auto"` | 导出模型的推理精度。可选值：`"auto"`、`"float16"`、`"float32"`、`"bfloat16"`。设为 `"auto"` 时，若原始模型为 fp32 且硬件支持 bf16，则自动转换为 bf16。 |
| `export_legacy_format` | `bool` | `false` | 是否使用旧格式（PyTorch `.bin`）导出。默认使用 safetensors 格式。 |

**训练配置示例：**

```yaml
peft_config:
  name: lora
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: all
  use_rslora: false
  use_dora: false
```

**断点续训示例：**

```yaml
peft_config:
  name: lora
  adapter_name_or_path: ./outputs/previous_lora_checkpoint
```

**导出合并示例：**

```yaml
peft_config:
  name: lora
  adapter_name_or_path: ./outputs/test_lora
  export_dir: ./merge_lora_model
  export_size: 5
  infer_dtype: auto
```

#### Freeze（部分参数冻结训练）

冻结模型大部分参数，仅训练指定层的参数。

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `name` | `str` | **必填** | 固定为 `"freeze"`。 |
| `freeze_trainable_layers` | `int` | `2` | 可训练的层数。正数表示训练模型**最后** N 层；负数表示训练模型**最前** N 层。 |
| `freeze_trainable_modules` | `str` 或 `list[str]` | `["all"]` | 在可训练层中，哪些模块参与训练。设为 `"all"` 表示可训练层中的所有模块都参与训练。也可以指定具体的模块名称。 |
| `freeze_extra_modules` | `list[str]` | `[]` | 额外需要训练的非隐藏层模块，例如 `embed_tokens`、`lm_head` 等。 |
| `cast_trainable_params_to_fp32` | `bool` | `true` | 是否将可训练参数转换为 fp32 精度。设为 `true` 可提高训练稳定性。 |

**配置示例：**

```yaml
peft_config:
  name: freeze
  freeze_trainable_layers: 2
  freeze_trainable_modules: all
  freeze_extra_modules: null
```

### 算子优化配置（kernel_config）

用于配置自定义高性能算子（Kernel），以加速模型的训练和推理。

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `name` | `str` | **必填** | 固定为 `"auto"`。 |
| `include_kernels` | `str` | `None` | 指定要应用的算子。可选值：`null`/`false`（不使用任何算子）、`"auto"`/`true`（使用所有默认注册的算子）、逗号分隔的算子 ID 列表（如 `"kernel_id1,kernel_id2"`，仅使用指定的算子）。 |

**配置示例：**

```yaml
kernel_config:
  name: auto
  include_kernels: auto
```

### 量化配置（quant_config）

用于配置模型量化，降低模型显存占用。目前支持通过 BitsAndBytes 进行 4-bit 和 8-bit 量化。

#### 自动量化（auto）

根据配置自动选择量化方法（目前默认使用 BitsAndBytes）。

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `name` | `str` | **必填** | 设为 `"auto"` 使用自动选择，或设为 `"bnb"` 直接指定使用 BitsAndBytes。 |
| `quantization_bit` | `int` | `None` | 量化位数。可选值：`4`（4-bit 量化）、`8`（8-bit 量化）。 |

#### BitsAndBytes（bnb）

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `name` | `str` | **必填** | 固定为 `"bnb"`。 |
| `quantization_bit` | `int` | `4` | 量化位数。可选值：`4`、`8`。 |
| `compute_dtype` | `str` | `"float16"` | 计算精度。量化模型在计算时使用的浮点精度。 |
| `double_quantization` | `bool` | `true` | 是否启用双重量化（Double Quantization）。可进一步减少显存占用。仅 4-bit 量化时有效。 |
| `quantization_type` | `str` | `"nf4"` | 量化类型。可选值：`"nf4"`（NormalFloat4，推荐）、`"fp4"`。仅 4-bit 量化时有效。 |

**配置示例（QLoRA 4-bit）：**

```yaml
quant_config:
  name: bnb
  quantization_bit: 4

peft_config:
  name: lora
  r: 16
  lora_alpha: 32
  target_modules: all
```

> [!NOTE]
> 使用 4-bit 量化 + LoRA 的组合即为 QLoRA 训练方式，这是目前最节省显存的微调方案之一。
>
> 使用 8-bit 量化仅支持推理，不支持训练。训练仅支持 4-bit 量化（配合 FSDP + QLoRA）。
