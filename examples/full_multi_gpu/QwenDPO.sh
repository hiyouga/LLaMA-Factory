#!/bin/bash
export WANDB_DISABLED=true
wandb offline

deepspeed --num_gpus 8 ../../src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path /mnt/data/shesj/Trained/RL4CoT/SFT/Qwen1_5_7b_Chat_SixLang \
    --dataset orca_rlhf \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../saves/LLaMA2-7B/lora/dpo \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 80 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.03 \
    --save_steps 200 \
    --eval_steps 200 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --max_samples 10000000000 \
    --val_size 0.1 \
    --dpo_ftx 1.0 \
    --bf16 \
    --tf32 True