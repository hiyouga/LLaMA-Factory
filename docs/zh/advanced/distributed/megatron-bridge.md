# Megatron Bridge

[Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/) 是 NVIDIA 提供的 Hugging Face ↔ Megatron-Core 桥梁。LLaMA-Factory 用它走独立的 PT / SFT 训练路径，而不是 Hugging Face Trainer。

启用方式：设置环境变量 `USE_MEGATRON_BRIDGE=1`。启动器会强制 `FORCE_TORCHRUN=1`。

> 这与 `USE_MCA=1`（[mcore_adapter](https://github.com/alibaba/ROLL/tree/main/mcore_adapter)）是两条后端，不要同时开启。

## 当前范围

| 项目 | 支持情况 |
| --- | --- |
| 训练阶段 | `pt`、`sft` |
| 微调方式 | `full`、`lora` |
| 量化 / QLoRA | 不支持 |
| DeepSpeed / MCA / HyperParallel | 互斥，不能同时开启 |
| Trainer callback | 暂不支持，传入会被忽略 |
| 多模态 / 音频 / Omni | v0 未启用 |

支持的 Hugging Face `model_type`：

`deepseek_v3`、`deepseek_v4`、`llama`、`mistral`、`qwen2`、`qwen3`、`qwen3_5`、`qwen3_5_moe`、`qwen3_5_moe_text`、`qwen3_5_text`、`qwen3_moe`、`qwen3_next`

未在上表中的模型（包括 VL / Omni）会在启动时报错。

## 安装

推荐先装 PyTorch 与 [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)，再安装 Megatron Bridge：

```bash
pip install --no-build-isolation transformer-engine[pytorch]
pip install --no-build-isolation megatron-bridge
```

也可以使用 [NeMo Framework 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)，或构建本仓库的 CUDA 镜像：

```bash
docker build -f docker/docker-cuda/Dockerfile.mbridge \
  -t llamafactory-megatron-bridge:latest .
```

验证导入：

```bash
python - <<'PY'
from megatron.bridge import AutoBridge
print("megatron-bridge import ok")
PY
```

APEX 的 `fused_weight_gradient_mlp_cuda` 是可选依赖。未安装时，LLaMA-Factory 会自动关闭 `gradient_accumulation_fusion`，训练仍可继续。

## 快速开始

在仓库根目录执行：

```bash
USE_MEGATRON_BRIDGE=1 llamafactory-cli train examples/megatron_bridge/llama3_sft.yaml
```

多卡示例（8 GPU，TP=2）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 USE_MEGATRON_BRIDGE=1 \
  llamafactory-cli train examples/megatron_bridge/llama3_sft.yaml \
  tensor_model_parallel_size=2 \
  sequence_parallel=true
```

多机与普通 `llamafactory-cli train` 相同，继续使用 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。

完整配置见 `examples/megatron_bridge/llama3_sft.yaml`。

## 并行与批大小

并行度必须整除 `world_size`：

```text
TP * PP * CP * EP  <= world_size
world_size % (TP * PP * CP * EP) == 0
DP = world_size / (TP * PP * CP * EP)
```

`sequence_parallel=true` 要求 `tensor_model_parallel_size > 1`。

全局 batch 按 Megatron 语义计算：

```text
global_batch_size = per_device_train_batch_size
                    * gradient_accumulation_steps
                    * DP
```

训练步数：`max_steps > 0` 时直接作为 `train_iters`；否则按 `num_train_epochs` 和样本数推算。

`context_parallel_size > 1` 时，SFT 会强制 `calculate_per_token_loss=true`。若同时开启 `use_packed_sequences`，packed 序列会按 `CP * 2` 对齐。

## 配置字段

Megatron Bridge 专用参数定义在 `MegatronBridgeArguments`，与普通训练 YAML 写在一起即可。

### 并行

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `tensor_model_parallel_size` | `1` | 张量并行 TP |
| `pipeline_model_parallel_size` | `1` | 流水线并行 PP |
| `expert_model_parallel_size` | `1` | MoE 专家并行 EP |
| `context_parallel_size` | `1` | 上下文并行 CP |
| `virtual_pipeline_model_parallel_size` | `None` | 交错流水线 VPP |
| `sequence_parallel` | `false` | 序列并行，需 TP > 1 |

### 重计算与融合

| 字段 | 说明 |
| --- | --- |
| `recompute_granularity` | `full` 或 `selective` |
| `recompute_method` | `uniform` 或 `block` |
| `recompute_num_layers` | 每个重计算单元的层数 |
| `account_for_embedding_in_pipeline_split` | PP 切分是否计入 embedding |
| `account_for_loss_in_pipeline_split` | PP 切分是否计入 loss |
| `bias_activation_fusion` / `apply_rope_fusion` / `masked_softmax_fusion` / `cross_entropy_loss_fusion` | 设为 `None` 时保留 Megatron provider 默认值 |

### 优化器与精度

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `use_distributed_optimizer` | `true` | Megatron distributed optimizer |
| `overlap_param_gather` | `true` | 参数 all-gather 与前向重叠 |
| `overlap_grad_reduce` | `true` | 梯度 all-reduce 与反向重叠 |
| `mixed_precision` | `bf16_mixed` | 如 `bf16_mixed`、`fp8` |
| `moe_grouped_gemm` | `None` | MoE grouped GEMM |
| `moe_token_dispatcher_type` | `None` | `allgather` / `alltoall` / `flex` |

学习率、warmup、`adam_beta1` / `adam_beta2`、`weight_decay`、`max_grad_norm` 仍使用标准 `TrainingArguments`。`lr_scheduler_type` 仅映射 `cosine`、`linear`、`constant`、`constant_with_warmup`；其余会回退到 cosine。`full` / `lora` 的 `min_lr` 为 `0.0`。

### 数据与 checkpoint

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `use_packed_sequences` | `false` | SFT packed sequence |
| `megatron_pretrained_checkpoint` | `None` | 已有 Megatron 格式权重；未设置时自动从 HF 转换 |
| `export_hf_on_finish` | `false` | 训练结束后导出 Hugging Face 权重 |
| `extra_config` | `None` | JSON 字符串或 JSON 文件，支持点路径覆盖 |

```yaml
extra_config: '{"train.train_iters": 5, "logger.log_interval": 1}'
# 或
# extra_config: /path/to/overrides.json
```

## 数据流

LLaMA-Factory 仍负责数据集加载与 `template`。训练开始前，rank 0 会把对齐后的样本导出到 `output_dir/mb_dataset/`：

| 阶段 | 文件 | 格式 |
| --- | --- | --- |
| `pt` | `training.jsonl` | `{"text": "..."}` |
| `sft` | `training.jsonl`、可选 `validation.jsonl` | 优先 Hugging Face `messages`；无法注入 `{% generation %}` 时回退 ShareGPT |

SFT 会尽量使用与 Hugging Face Trainer 相同的 chat template，并为 assistant 回复注入 `{% generation %}`，以便只对回答计算 loss。

## Checkpoint

SFT 首次运行会把 Hugging Face 权重转换成 Megatron 格式，默认目录为 `output_dir/megatron_pretrained/`。已转换过的目录会复用。若已有 Megatron checkpoint，直接设置 `megatron_pretrained_checkpoint`。

训练过程写入 Megatron distributed checkpoint（`torch_dist`）：

```text
output_dir/
  latest_checkpointed_iteration.txt
  iter_XXXXXXX/
    run_config.yaml
    ...
  mb_dataset/
  megatron_pretrained/     # SFT 自动转换
  hf_export/               # export_hf_on_finish=true
```

续训：保持相同 `output_dir`，并设置 `overwrite_output_dir: false`。存在 `latest_checkpointed_iteration.txt` 或 `latest_train_state.pt` 时会从该目录恢复。不要依赖 Hugging Face 的 `resume_from_checkpoint` 路径语义。

`export_hf_on_finish: true` 时：

- 全参数：导出完整 Hugging Face 目录到 `output_dir/hf_export/`
- LoRA：只导出 PEFT adapter（Megatron PEFT checkpoint 不含基座权重）

短跑、只对比 loss 时建议关掉导出，避免额外 checkpoint 显存开销。

## LoRA

```yaml
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
```

Megatron Bridge 固定作用在 `linear_qkv`、`linear_proj`、`linear_fc1`、`linear_fc2`。自定义 `lora_target` 会被忽略并打 warning。`lora_rank` / `lora_alpha` 会映射到 Megatron LoRA 的 `dim` / `alpha`；`lora_dropout` 不会传入。

## 与 MCA 的区别

| | Megatron Bridge | MCA (`mcore_adapter`) |
| --- | --- | --- |
| 环境变量 | `USE_MEGATRON_BRIDGE=1` | `USE_MCA=1` |
| 依赖 | `megatron-bridge` | `mcore-adapter` |
| 阶段 | `pt`、`sft` | `pt`、`sft`、`dpo` |
| 微调 | `full`、`lora` | 目前以 `full` 为主 |
| 示例 | `examples/megatron_bridge/` | `examples/megatron/` |
| Docker | `Dockerfile.mbridge` | `Dockerfile.megatron` |

## 限制与排障

- 不要同时设置 `USE_MEGATRON_BRIDGE` 和 `USE_MCA`。
- 不要搭配 DeepSpeed、量化模型或 Hugging Face Trainer callback。
- 部分 GPU（例如 V100）保存 distributed optimizer 时，异步 D2H copy 可能失败；LLaMA-Factory 会改用同步拷贝。
- pip 安装的 `megatron-core` 若已带 `helpers_cpp`，会跳过源码树里的 `make` 编译。
- Web UI 可以提交同一份 YAML，但仍需在启动环境中设置 `USE_MEGATRON_BRIDGE=1`。

更多上游概念见 [Megatron Bridge 文档](https://docs.nvidia.com/nemo/megatron-bridge/latest/)。
