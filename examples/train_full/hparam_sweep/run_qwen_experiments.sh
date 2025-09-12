#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B full SFT hparam sweep (single node, 8xH100)
# Edit INVOCATION to match your LLaMA-Factory launcher.

NGPUS=${NGPUS:-8}
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_ROOT="$ROOT_DIR/runs"
mkdir -p "$LOG_ROOT"

CONFIGS=(
  "$ROOT_DIR/../qwen3_full_sft_deepspeed_online.yaml"               # baseline
  "$ROOT_DIR/qwen3_full_sft_e1_adamw_torch_betas095.yaml"
  "$ROOT_DIR/qwen3_full_sft_e2_adamw_bnb_8bit.yaml"
  "$ROOT_DIR/qwen3_full_sft_e3_paged_adamw_8bit.yaml"
  "$ROOT_DIR/qwen3_full_sft_e4_dataloader16.yaml"
  "$ROOT_DIR/qwen3_full_sft_e5_group_by_length.yaml"
  "$ROOT_DIR/qwen3_full_sft_e6_grad_ckpt_off.yaml"
  "$ROOT_DIR/qwen3_full_sft_e7_warmup_0p10.yaml"
  "$ROOT_DIR/qwen3_full_sft_e8_label_smoothing_0p05.yaml"
  "$ROOT_DIR/qwen3_full_sft_e9_max_grad_norm_5e-5.yaml"
  "$ROOT_DIR/qwen3_full_sft_e10_adamw_torch_eps5e-9.yaml"
  "$ROOT_DIR/qwen3_full_sft_lr_1e-6.yaml"
  "$ROOT_DIR/qwen3_full_sft_lr_3e-6.yaml"
  "$ROOT_DIR/qwen3_full_sft_lr_5e-6.yaml"
  "$ROOT_DIR/qwen3_full_sft_sched_linear.yaml"
  "$ROOT_DIR/qwen3_full_sft_sched_constant_warmup.yaml"
  "$ROOT_DIR/qwen3_full_sft_opt_adamw_torch_fused.yaml"
  "$ROOT_DIR/qwen3_full_sft_opt_adamw_hf.yaml"
  "$ROOT_DIR/qwen3_full_sft_opt_adafactor.yaml"
  "$ROOT_DIR/qwen3_full_sft_opt_muon.yaml"
  "$ROOT_DIR/qwen3_full_sft_opt_adam_mini.yaml"
  "$ROOT_DIR/qwen3_full_sft_opt_badam_layer.yaml"
)

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }

# Pick the invocation you use; comment the other.
# A) LLaMA-Factory CLI with config file
# run_train() { llamafactory-cli train --config_file "$1"; }

# B) Python entry
# run_train() { python -m llamafactory.train --config_file "$1"; }

# C) Or inline HF args (discouraged here since we are sweeping YAMLs)
# run_train() { llamafactory-cli train --deepspeed examples/deepspeed/ds_z3_config.json ...; }

for CFG in "${CONFIGS[@]}"; do
  NAME=$(basename "$CFG" .yaml)
  OUT_DIR="$LOG_ROOT/$NAME"
  mkdir -p "$OUT_DIR"
  LOG_FILE="$OUT_DIR/train_$(timestamp).log"

  echo "==> Running $NAME"
  # Uncomment the run_train line matching your launcher
  # run_train "$CFG" 2>&1 | tee "$LOG_FILE"
  echo "Completed $NAME. Log: $LOG_FILE"
done

echo "All Qwen experiments queued. Logs under: $LOG_ROOT"
