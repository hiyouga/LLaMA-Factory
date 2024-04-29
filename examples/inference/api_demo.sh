#!/bin/bash

CUDA_VISIBLE_DEVICES=0 API_PORT=8000 python ../../src/api_demo.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \
    --model_revision Llama-2-7b \
    --template default \
    --finetuning_type lora
