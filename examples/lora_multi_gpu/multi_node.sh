#!/bin/bash
# also launch it on slave machine using slave_config.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file examples/accelerate/master_config.yaml \
    src/train.py examples/lora_multi_gpu/llama3_lora_sft.yaml
