#!/bin/bash
# add `--visual_inputs True` to load MLLM

CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \
    --template default \
    --finetuning_type lora
