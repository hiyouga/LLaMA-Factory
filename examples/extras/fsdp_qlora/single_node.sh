#!/bin/bash
# DO NOT use GPTQ/AWQ model in FSDP+QLoRA

pip install "transformers>=4.39.1"
pip install "accelerate>=0.28.0"
pip install "bitsandbytes>=0.43.0"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/extras/fsdp_qlora/llama3_lora_sft.yaml
