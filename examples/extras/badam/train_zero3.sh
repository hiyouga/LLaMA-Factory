#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ../../..

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset alpaca_en_demo \
    --cutoff_len 1024 \
    --learning_rate 1e-6 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --use_badam True \
    --output_dir saves/LLaMA2-13B/full/BAdam \
    --fp16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --badam_mode layer \
    --badam_switch_mode ascending \
    --badam_switch_interval 50 \
    --deepspeed cache/ds_z3_config.json 