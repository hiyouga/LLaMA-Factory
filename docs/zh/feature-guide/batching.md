# 批处理策略

`batching_strategy` 决定样本如何组成 micro-batch。

| 策略 | 行为 |
|------|------|
| `normal` | 固定样本数，按最长序列 padding |
| `padding_free` | 固定样本数，拼接序列以移除 padding |
| `dynamic_batching` | 按 token 预算动态决定 batch 大小 |
| `dynamic_padding_free` | 动态 batch 并拼接序列 |

## 选择批处理策略

```yaml
batching_strategy: dynamic_padding_free
micro_batch_size: 4
cutoff_len: 2048
max_steps: 100
flash_attn: flash_attention_2
```

仓库在 `examples/v1/train_batching_strategy/` 下为四种策略提供了完整示例。

## 动态批处理约束

- `dynamic_batching` 必须设置正数 `max_steps`。
- `dynamic_batching` 不支持 `save_epochs`，应使用 `save_steps`。
- padding-free 示例使用 `flash_attention_2`。
- `micro_batch_size` 对动态策略表示构造批次时使用的基础预算，最终样本
  数由 token 数动态确定。

内部 collate 和状态恢复流程见
[BatchGenerator](../developer-guide/core/batch_generator.md)。
