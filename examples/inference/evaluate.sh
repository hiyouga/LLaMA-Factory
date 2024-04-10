#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../src/evaluate.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \
    --template vanilla \
    --finetuning_type lora \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 4
