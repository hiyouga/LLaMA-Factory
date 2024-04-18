#!/bin/bash
export WANDB_DISABLED=true
wandb offline

deepspeed --num_gpus 8 ../../src/train_bash.py \
    --deepspeed ../deepspeed/ds_z2_config.json \
    --stage dpo \
    --do_train \
    --model_name_or_path /mnt/data/shesj/Trained/RL4CoT/SFT/Mistral-7B_SixLang \
    --dataset sixlan_align_data_5k_MisConInstructLowSix_0.7_20-4ensemble-train \
    --dataset_dir /mnt/data/shesj/Data/LFData \
    --template mistral \
    --finetuning_type full \
    --output_dir /mnt/data/shesj/Trained/CommonAlign/DPO/MistralSixDPO \
    --logging_dir "/mnt/data/shesj/Log/CommonAlign/DPO/MistralSixDPO" \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 160 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.03 \
    --save_steps 200 \
    --eval_steps 200 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 1e-6 \
    --max_steps 1000 \
    --max_samples 10000000000 \
    --val_size -1 \
    --dpo_ftx 1.0 \
    --bf16 \
    --tf32 True