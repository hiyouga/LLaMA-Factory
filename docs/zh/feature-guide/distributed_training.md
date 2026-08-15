# 分布式训练

训练命令检测到多设备后会自动通过 `torchrun` 启动。拓扑字段属于
`TrainingArguments`，后端专属字段放在 `dist_config`。

## FSDP2

```yaml
dist_config:
  name: fsdp2
  reshard_after_forward: true
  offload_params: false
  pin_memory: true
  dcp_path: null
```

## FSDPTurbo

FSDPTurbo 在 FSDP2 基础上提供 MoE 专家并行和专家参数分片。先安装对应
依赖：

```bash
python -m pip install -r requirements/fsdpturbo.txt
```

然后在 `dist_config` 中选择 `fsdpturbo`：

```yaml
dist_config:
  name: fsdpturbo
  ep_size: 16
  ep_dispatcher: eager
```

`ep_size` 必须能够整除 data parallel size。完整示例见
`examples/v1/train_full/train_full_qwen3_moe_fsdpturbo_ep_fsdp.yaml`。

## DeepSpeed

```yaml
dist_config:
  name: deepspeed
  config_file: examples/deepspeed/ds_z3_config.json
```

`config_file` 是必填字段。

## Ulysses Context Parallel

```yaml
flash_attn: flash_attention_2
cp_mode: ulysses
cp_size: 2

dist_config:
  name: fsdp2
```

## 配置并行维度

```yaml
dp_size: null
cp_size: 1
cp_mode: ulysses
mp_replicate_size: 1
mp_shard_size: null
dist_timeout: 18000
```

未显式指定时，`dp_size` 和 `mp_shard_size` 根据 world size 推导。
`mp_replicate_size` 是模型复制的份数，`mp_shard_size` 是参数切分的份数，
两者乘积等于 world size。后端完整配置见
[训练参数](../configuration/training.md#dist_config)。

## 配置多机启动

CLI 读取 `NNODES`、`NODE_RANK`、`NPROC_PER_NODE`、`MASTER_ADDR` 和
`MASTER_PORT`。设置 `RDZV_ID` 时启用 elastic rendezvous，并可配合
`MIN_NNODES`、`MAX_NNODES`、`MAX_RESTARTS`。
