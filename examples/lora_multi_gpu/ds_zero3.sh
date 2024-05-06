#!/bin/bash
# ZeRO-3 enables weight sharding on multiple GPUs

deepspeed --include "localhost:0,1,2,3" \
    src/train.py examples/lora_multi_gpu/llama3_lora_sft_ds.yaml
