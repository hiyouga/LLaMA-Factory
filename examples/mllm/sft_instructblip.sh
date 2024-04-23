#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft_mm \
    --do_train \
    --model_name_or_path /home/LAB/fengzc/LLM/checkpoints/Salesforce/instructblip-vicuna-7b \
    --dataset llava_instruct_100 \
    --dataset_dir data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,k_proj \
    --output_dir saves/instructblip-vicuna-7b/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 3000 \
    --val_size 0.1 \
    --plot_loss \
    --quantization_bit 8 \
    --image_path /home/LAB/fengzc/LLM/checkpoints/liuhaotian/LLaVA-Instruct-150K/images/coco/train2017 \
    --use_qformer

