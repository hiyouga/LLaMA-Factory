#!/bin/bash

deepspeed --include "localhost:0,1,2,3" \
    src/train.py examples/full_multi_gpu/llama3_full_sft.yaml
