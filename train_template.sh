#!/bin/bash

set -e

# Check for the correct number of arguments
if [ "$#" -ne 9 ]; then
    echo "Usage: $0 <device> <port> <model> <dataset> <template> <learning_rate> <batch_size> <gradient_accumulation_steps> <epoch>"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Assigning inputs to variables for clarity
DEVICES="$1"
PORT="$2"
MODEL="$3"
MODEL_NAME_FOR_LOG=$(basename $MODEL)
DATASET="$4"
TEMPLATE="$5"
LEARNING_RATE="$6"
BSZ="$7"
GAS="$8"
EPOCH="$9"

# total BSZ
NUM_DEVICES=$(echo "$DEVICES" | awk -F, '{print NF}')
TOTAL_BSZ=$((BSZ * GAS * NUM_DEVICES))

# Check and create dataset specific directories in models and logs
mkdir -p "/home/data/ch/targeted_IT/logs/model_training/${DATASET}"

COMMAND="nohup deepspeed --include localhost:${DEVICES} --master_port=${PORT} src/train_bash.py \
            --deepspeed /home/chenhao/projects/targeted_IT/LLaMA-Factory/ds.config.json \
            --stage sft \
            --model_name_or_path ${MODEL} \
            --do_train \
            --dataset ${DATASET} \
            --template ${TEMPLATE} \
            --finetuning_type full \
            --output_dir /home/data/ch/targeted_IT/models/${DATASET}/${MODEL_NAME_FOR_LOG}_bsz${TOTAL_BSZ}_lr${LEARNING_RATE}_e${EPOCH}_${TIMESTAMP} \
            --overwrite_cache \
            --per_device_train_batch_size ${BSZ} \
            --gradient_accumulation_steps ${GAS} \
            --lr_scheduler_type cosine \
            --logging_steps 10 \
            --learning_rate ${LEARNING_RATE} \
            --num_train_epochs ${EPOCH} \
            --plot_loss \
            --fp16 \
            --gradient_checkpointing True \
            --save_steps 10000000 \
            --use_fast_tokenizer"

# Define log path
LOG_PATH="/home/data/ch/targeted_IT/logs/model_training/${DATASET}/${MODEL_NAME_FOR_LOG}_bsz${TOTAL_BSZ}_lr${LEARNING_RATE}_e${EPOCH}_${TIMESTAMP}.log"

echo "train.sh started. Check ${LOG_PATH} for logs." | tee -a "$LOG_PATH"
echo "Command: $COMMAND" | tee -a "$LOG_PATH"

# Execute the command and redirect all outputs to the log file
eval $COMMAND >> "$LOG_PATH" 2>&1
