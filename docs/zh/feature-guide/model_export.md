# 模型导出

`merge` 命令将一个或多个 LoRA adapter 依次合并到基座模型，并保存为
Hugging Face 格式目录。

## 导出配置

```yaml
model: Qwen/Qwen3-4B
peft_config:
  name: lora
  adapter_name_or_path: outputs/qwen3_lora
  export_dir: outputs/qwen3_merged
  export_size: 5
  infer_dtype: auto
  export_legacy_format: false
```

```bash
llamafactory-cli merge config.yaml
```

`export_size` 的单位为 GB。`infer_dtype` 支持 `auto`、`float16`、
`float32` 和 `bfloat16`。完整字段见
[模型参数](../configuration/model.md#peft_config)。
