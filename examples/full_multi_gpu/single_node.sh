#!/bin/bash

NPROC_PER_NODE=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --standalone \
    src/train.py examples/full_multi_gpu/llama3_full_sft.yaml
