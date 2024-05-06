#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \
    --template default \
    --finetuning_type lora \
    --export_dir ../../models/llama2-7b-sft \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False
