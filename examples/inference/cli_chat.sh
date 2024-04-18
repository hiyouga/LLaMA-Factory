#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../src/cli_demo.py \
    --model_name_or_path /mnt/data/shesj/Trained/CommonAlign/DPO/QwenSixDPO \
    --template qwen \
    --finetuning_type full