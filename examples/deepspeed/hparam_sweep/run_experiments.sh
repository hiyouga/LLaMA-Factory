#!/usr/bin/env bash
set -euo pipefail

# Basic usage:
#   1) Edit TRAIN_CMD section to match your launcher and training script/args.
#   2) Run:  bash run_experiments.sh
# Logs will be written under ./ds_runs/<config_basename>/

NGPUS=${NGPUS:-8}
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_ROOT="$ROOT_DIR/ds_runs"
mkdir -p "$LOG_ROOT"

# Choose which configs to run (ordered by priority)
CONFIGS=(
  "$ROOT_DIR/ds_z3_base_profile.json"
  "$ROOT_DIR/ds_z3_e1_bf16.json"
  "$ROOT_DIR/ds_z3_e2_overlap_rs.json"
  "$ROOT_DIR/ds_z3_e3_buckets_200M.json"
  "$ROOT_DIR/ds_z3_e4_autotune.json"
  "$ROOT_DIR/ds_z3_e5_persist_5e5.json"
  "$ROOT_DIR/ds_z3_e6_modgran_16.json"
  "$ROOT_DIR/ds_z3_e7_gradaccum_bf16.json"
  "$ROOT_DIR/ds_z3_e9_quant_grads.json"
  "$ROOT_DIR/ds_z3_e10_allgather_partitions_false.json"
  "$ROOT_DIR/ds_z3_e11_act_ckpt.json"
)

# EDIT THIS: Provide your training invocation.
# Option A) Deepspeed launcher + training script
# TRAIN_SCRIPT="/path/to/train.py"
# TRAIN_ARGS=(
#   --model_name_or_path /path/to/model
#   --data_path /path/to/dataset
#   --output_dir /path/to/outputs
#   --deepspeed  # will be followed by the config path
#   # ... other args (seq length 32768, etc.)
# )

# Option B) LLaMA-Factory CLI (uncomment and edit)
# TRAIN_CMD_PREFIX=(llamafactory-cli train)
# TRAIN_ARGS=(
#   --deepspeed  # will be followed by the config path
#   --model_name_or_path /path/to/model
#   --dataset /path/to/dataset
#   --output_dir /path/to/outputs
#   --cutoff_len 32768
#   # ... other args
# )

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }

for CFG in "${CONFIGS[@]}"; do
  NAME=$(basename "$CFG" .json)
  OUT_DIR="$LOG_ROOT/$NAME"
  mkdir -p "$OUT_DIR"
  LOG_FILE="$OUT_DIR/train_$(timestamp).log"

  echo "==> Running $NAME with $NGPUS GPUs"

  # Pick the launcher you use and comment out the other.

  # A) Deepspeed launcher (edit TRAIN_SCRIPT/ARGS above)
  # deepspeed --num_gpus "$NGPUS" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}" --deepspeed "$CFG" \
  #   2>&1 | tee "$LOG_FILE"

  # B) LLaMA-Factory CLI (edit TRAIN_ARGS above)
  # "${TRAIN_CMD_PREFIX[@]}" "${TRAIN_ARGS[@]}" "$CFG" \
  #   2>&1 | tee "$LOG_FILE"

  echo "Completed $NAME. Log: $LOG_FILE"
done

echo "All experiments queued. Logs under: $LOG_ROOT"
