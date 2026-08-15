# 模型保存与恢复

训练结束时 `save_model()` 保存最终模型；训练过程中可以按 step 或 epoch
保存 checkpoint。

## 保存训练 Checkpoint

```yaml
save_steps: 500
save_epochs: null
save_total_limit: 3
save_ckpt_as_hf: false
```

`save_steps` 与 `save_epochs` 控制触发时机。`save_total_limit` 删除最旧的
完整 checkpoint。

## 从 Checkpoint 恢复训练

```yaml
resume_from_checkpoint: auto
```

`auto` 在 `output_dir` 下寻找最新的完整 checkpoint，也可以直接填写
checkpoint 路径。

## 不同后端的保存格式

- FSDP2 默认使用分布式 checkpoint。
- `save_ckpt_as_hf: true` 将中间 checkpoint 保存为 HF 格式，但会提高
  保存时内存占用。
- DeepSpeed 的 checkpoint 由对应后端实现管理。

`dist_config.dcp_path` 用于模型初始化阶段加载 DCP 权重，与恢复训练器、
优化器和批次状态的 `resume_from_checkpoint` 不是同一个功能。
