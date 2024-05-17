#!/bin/bash

NPROC_PER_NODE=4
NNODES=2
RANK=0
MASTER_ADDR=192.168.0.1
MASTER_PORT=29500

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py examples/full_multi_gpu/llama3_full_sft.yaml
