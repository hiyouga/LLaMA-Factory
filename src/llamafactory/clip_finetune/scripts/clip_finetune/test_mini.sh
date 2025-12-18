#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# custom config
# custom config
DATA=/home/wangjilong/data1
TRAINER=CLIP_Adapter
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
ACCEPT=$3
if [ -z $ACCEPT ]; then
    ACCEPT=0
fi
train_directory_path="/home/wangjilong/data/mini-imagenet/train/"
new_train_directory_path="/home/wangjilong/data1/mini-imagenet/train/"
x=(234 117 58 234 117 58 234 117 58)
for i in 1 2 3 4 5 6 7 8 9
do

    CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/clip_finetune/${CFG}_${i}.yaml \
    --output-dir output/${TRAINER}_${i}_mini/${CFG}/${DATASET} \
    TRAINER.COOP.ACC $ACCEPT \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC True \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS 0 \
    TRAINER.COOP.Max_Batch ${x[i-1]}
done
