#!/bin/bash

python ../../../scripts/llama_pro.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir ../../../models/llama2-7b-pro \
    --num_expand 8
