#!/bin/bash

python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py examples/full_multi_gpu/llama3_full_sft.yaml
