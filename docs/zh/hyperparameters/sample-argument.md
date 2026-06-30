# 采样参数（SampleArguments）

采样参数用于控制模型在推理和评估阶段的生成行为，包括采样后端的选择和生成长度限制。

## 参数列表

| 参数名 | 类型 | 默认值 | 说明 |
|:---:|:---:|:---:|:---|
| `sample_backend` | `str` | `"hf"` | 采样后端。可选值见下方[采样后端](#采样后端)。 |
| `max_new_tokens` | `int` | `128` | 最大生成 token 数。控制模型在推理/评估时生成文本的最大长度。 |

## 采样后端

| 可选值 | 说明 |
|:---:|:---|
| `hf` | 使用 HuggingFace Transformers 原生的 `generate()` 方法进行采样。兼容性最好，适合调试和小规模推理。 |
| `vllm` | 使用 vLLM 高性能推理引擎进行采样。支持 PagedAttention、连续批处理等优化，适合大规模推理场景。需额外安装 vLLM。 |

## 配置示例

```yaml
sample_backend: hf
max_new_tokens: 256
```
