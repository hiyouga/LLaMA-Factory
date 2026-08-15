# 推理

v1 提供 `chat` 入口。模型对话格式来自 tokenizer 自带的 Hugging Face
chat template；没有模板时回退到内置 ChatML。

## 启动 CLI 对话

```yaml
model: Qwen/Qwen3-4B-Instruct-2507
sample_backend: hf
max_new_tokens: 512
```

```bash
llamafactory-cli chat config.yaml
```

## 覆盖模型 Chat Template

`custom_chat_template` 接收一段 Jinja2 模板字符串，并覆盖 tokenizer
自带模板：

```yaml
model: path/to/model
custom_chat_template: >-
  {% for message in messages %}
  {{ message['role'] + ': ' + message['content'] }}
  {% endfor %}
```

这不是旧版的模板名称注册机制；v1 不再接受 `template: <name>` 字段。

## 使用 LoRA Adapter

```yaml
peft_config:
  name: lora
  adapter_name_or_path: outputs/qwen3_lora
```

推理模式会依次合并 `adapter_name_or_path` 中的 adapter。需要生成独立
HF 模型目录时使用[模型导出](model_export.md)。

## 选择采样后端

CLI 通过 `sample_backend` 选择采样后端。使用 Hugging Face 采样时设置：

```yaml
sample_backend: hf
```

完整字段见[推理参数](../configuration/inference.md)。
