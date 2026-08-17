# Megatron Bridge

[Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/) is NVIDIA's Hugging Face ↔ Megatron-Core bridge. LLaMA-Factory uses it as a standalone PT / SFT path outside the Hugging Face Trainer.

Enable it with `USE_MEGATRON_BRIDGE=1`. The launcher then forces `FORCE_TORCHRUN=1`.

> This is a different backend from `USE_MCA=1` ([mcore_adapter](https://github.com/alibaba/ROLL/tree/main/mcore_adapter)). Do not enable both.

## Current scope

| Item | Support |
| --- | --- |
| Stages | `pt`, `sft` |
| Finetuning | `full`, `lora` |
| Quantization / QLoRA | Not supported |
| DeepSpeed / MCA / HyperParallel | Mutually exclusive |
| Trainer callbacks | Not supported; ignored if provided |
| Multimodal / audio / Omni | Not enabled in v0 |

Supported Hugging Face `model_type` values:

`deepseek_v3`, `deepseek_v4`, `llama`, `mistral`, `qwen2`, `qwen3`, `qwen3_5`, `qwen3_5_moe`, `qwen3_5_moe_text`, `qwen3_5_text`, `qwen3_moe`, `qwen3_next`

Any other type, including VL / Omni models, fails at launch.

## Installation

Install PyTorch and [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) first, then Megatron Bridge:

```bash
pip install --no-build-isolation transformer-engine[pytorch]
pip install --no-build-isolation megatron-bridge
```

You can also use a [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags), or build the CUDA image in this repo:

```bash
docker build -f docker/docker-cuda/Dockerfile.mbridge \
  -t llamafactory-megatron-bridge:latest .
```

Sanity check:

```bash
python - <<'PY'
from megatron.bridge import AutoBridge
print("megatron-bridge import ok")
PY
```

The APEX extension `fused_weight_gradient_mlp_cuda` is optional. If it is missing, LLaMA-Factory disables `gradient_accumulation_fusion` automatically.

## Quick start

From the repository root:

```bash
USE_MEGATRON_BRIDGE=1 llamafactory-cli train examples/megatron_bridge/llama3_sft.yaml
```

Eight GPUs with TP=2:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 USE_MEGATRON_BRIDGE=1 \
  llamafactory-cli train examples/megatron_bridge/llama3_sft.yaml \
  tensor_model_parallel_size=2 \
  sequence_parallel=true
```

Multi-node launch is the same as a normal `llamafactory-cli train` job: set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`.

See `examples/megatron_bridge/llama3_sft.yaml` for a complete config.

## Parallelism and batch size

The product of parallel sizes must divide `world_size`:

```text
TP * PP * CP * EP  <= world_size
world_size % (TP * PP * CP * EP) == 0
DP = world_size / (TP * PP * CP * EP)
```

`sequence_parallel=true` requires `tensor_model_parallel_size > 1`.

Global batch size uses Megatron semantics:

```text
global_batch_size = per_device_train_batch_size
                    * gradient_accumulation_steps
                    * DP
```

If `max_steps > 0`, it becomes `train_iters`. Otherwise the schedule is derived from `num_train_epochs` and the dataset length.

When `context_parallel_size > 1`, SFT forces `calculate_per_token_loss=true`. Packed sequences, if enabled, are padded to a multiple of `CP * 2`.

## Configuration

Megatron Bridge options live in `MegatronBridgeArguments` and are written in the same training YAML.

### Parallelism

| Field | Default | Meaning |
| --- | --- | --- |
| `tensor_model_parallel_size` | `1` | Tensor parallel (TP) |
| `pipeline_model_parallel_size` | `1` | Pipeline parallel (PP) |
| `expert_model_parallel_size` | `1` | Expert parallel (EP) for MoE |
| `context_parallel_size` | `1` | Context parallel (CP) |
| `virtual_pipeline_model_parallel_size` | `None` | Interleaved virtual pipeline (VPP) |
| `sequence_parallel` | `false` | Sequence parallel; requires TP > 1 |

### Recompute and fusion

| Field | Meaning |
| --- | --- |
| `recompute_granularity` | `full` or `selective` |
| `recompute_method` | `uniform` or `block` |
| `recompute_num_layers` | Layers per recompute unit |
| `account_for_embedding_in_pipeline_split` | Include embedding in the PP split |
| `account_for_loss_in_pipeline_split` | Include loss in the PP split |
| `bias_activation_fusion` / `apply_rope_fusion` / `masked_softmax_fusion` / `cross_entropy_loss_fusion` | `None` keeps the Megatron provider default |

### Optimizer and precision

| Field | Default | Meaning |
| --- | --- | --- |
| `use_distributed_optimizer` | `true` | Megatron distributed optimizer |
| `overlap_param_gather` | `true` | Overlap parameter all-gather with forward |
| `overlap_grad_reduce` | `true` | Overlap gradient all-reduce with backward |
| `mixed_precision` | `bf16_mixed` | For example `bf16_mixed` or `fp8` |
| `moe_grouped_gemm` | `None` | Grouped GEMM for MoE experts |
| `moe_token_dispatcher_type` | `None` | `allgather`, `alltoall`, or `flex` |

Learning rate, warmup, `adam_beta1` / `adam_beta2`, `weight_decay`, and `max_grad_norm` still come from standard `TrainingArguments`. `lr_scheduler_type` maps `cosine`, `linear`, `constant`, and `constant_with_warmup`; other values fall back to cosine. For `full` / `lora`, `min_lr` is `0.0`.

### Data and checkpoints

| Field | Default | Meaning |
| --- | --- | --- |
| `use_packed_sequences` | `false` | Packed sequences for SFT |
| `megatron_pretrained_checkpoint` | `None` | Existing Megatron checkpoint; HF weights are converted if unset |
| `export_hf_on_finish` | `false` | Export Hugging Face weights after training |
| `extra_config` | `None` | JSON string or JSON file; dot-paths are supported |

```yaml
extra_config: '{"train.train_iters": 5, "logger.log_interval": 1}'
# or
# extra_config: /path/to/overrides.json
```

## Data flow

LLaMA-Factory still loads datasets and applies `template`. Rank 0 then exports aligned samples to `output_dir/mb_dataset/` before training:

| Stage | Files | Format |
| --- | --- | --- |
| `pt` | `training.jsonl` | `{"text": "..."}` |
| `sft` | `training.jsonl`, optional `validation.jsonl` | Hugging Face `messages` when possible; otherwise ShareGPT |

SFT prefers the same chat template as the Hugging Face Trainer and injects `{% generation %}` around assistant turns so loss is computed on responses only.

## Checkpoints

The first SFT run converts Hugging Face weights to Megatron format under `output_dir/megatron_pretrained/` and reuses that directory later. Set `megatron_pretrained_checkpoint` if you already have a Megatron checkpoint.

Training writes Megatron distributed checkpoints (`torch_dist`):

```text
output_dir/
  latest_checkpointed_iteration.txt
  iter_XXXXXXX/
    run_config.yaml
    ...
  mb_dataset/
  megatron_pretrained/     # automatic SFT conversion
  hf_export/               # when export_hf_on_finish=true
```

To resume, keep the same `output_dir` and set `overwrite_output_dir: false`. Training resumes when `latest_checkpointed_iteration.txt` or `latest_train_state.pt` exists. Do not rely on Hugging Face `resume_from_checkpoint` path semantics.

With `export_hf_on_finish: true`:

- Full finetuning exports a complete Hugging Face folder to `output_dir/hf_export/`
- LoRA exports a PEFT adapter only (Megatron PEFT checkpoints do not store base weights)

Disable export on short loss-comparison runs to avoid extra checkpoint memory pressure.

## LoRA

```yaml
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
```

Megatron Bridge always targets `linear_qkv`, `linear_proj`, `linear_fc1`, and `linear_fc2`. A custom `lora_target` is ignored with a warning. `lora_rank` / `lora_alpha` map to Megatron LoRA `dim` / `alpha`; `lora_dropout` is not forwarded.

## Difference from MCA

| | Megatron Bridge | MCA (`mcore_adapter`) |
| --- | --- | --- |
| Environment variable | `USE_MEGATRON_BRIDGE=1` | `USE_MCA=1` |
| Dependency | `megatron-bridge` | `mcore-adapter` |
| Stages | `pt`, `sft` | `pt`, `sft`, `dpo` |
| Finetuning | `full`, `lora` | Primarily `full` today |
| Examples | `examples/megatron_bridge/` | `examples/megatron/` |
| Docker | `Dockerfile.mbridge` | `Dockerfile.megatron` |

## Limitations

- Do not set `USE_MEGATRON_BRIDGE` and `USE_MCA` together.
- Do not combine this backend with DeepSpeed, quantized models, or Hugging Face Trainer callbacks.
- On some GPUs (for example V100), async D2H copies while saving the distributed optimizer can fail; LLaMA-Factory falls back to blocking copies.
- If pip-installed `megatron-core` already ships `helpers_cpp`, helper compilation via `make` is skipped.
- LlamaBoard can submit the same YAML, but the process environment still needs `USE_MEGATRON_BRIDGE=1`.

See the upstream [Megatron Bridge docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/) for Megatron-side concepts.
