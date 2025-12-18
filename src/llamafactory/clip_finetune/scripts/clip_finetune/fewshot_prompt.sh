#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# custom config
DATA=/home/wangjilong/data
TRAINER=CLIP_Prompt
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
PTL=$7
DEEP=$8
if [ -z $PTL ]; then
    PTL=4
fi
if [ -z $DEEP ]; then
    DEEP=falese
fi

CUDA_VISIBLE_DEVICES=0 python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/clip_finetune/${CFG}_prompt.yaml \
--output-dir output/${TRAINER}_${PTL}_${DEEP}/${CFG}/${DATASET} \
TRAINER.COOP.N_PLN $PTL \
TRAINER.COOP.PMT_DEEP $DEEP \
TRAINER.COOP.N_CTX ${NCTX} \
TRAINER.COOP.CSC ${CSC} \
TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS}
