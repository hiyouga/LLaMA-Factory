#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ../../saves/LLaMA2-7B/full/sft \
    --dataset alpaca_gpt4_en,glaive_toolcall \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type full \
    --output_dir ../../saves/LLaMA2-7B/full/predict \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 20 \
    --predict_with_generate
