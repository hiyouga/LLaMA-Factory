#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../src/export_model.py \
    --model_name_or_path ../../models/llama2-7b-sft \
    --template default \
    --export_dir ../../models/llama2-7b-sft-int4 \
    --export_quantization_bit 4 \
    --export_quantization_dataset ../../data/c4_demo.json \
    --export_size 2 \
    --export_legacy_format False
