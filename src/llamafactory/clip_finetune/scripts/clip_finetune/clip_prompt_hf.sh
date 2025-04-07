#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# custom config

TRAINER=CLIP_VPT_hf
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
PTL=$3
DEEP=$4
ACCEPT=$5
DEVICE=$6
USE_OPTUNA=$7
device=0
if [ -z $ACCEPT ]; then
    ACCEPT=0
fi
if [ -z $PTL ]; then
    PTL=4
fi
if [ -z $DEEP ]; then
    DEEP=falese
fi
if [ -z $DEVICE ]; then
    DEVICE="cuda"
    device=0
fi
if [ $DEVICE = "XPU" ]; then
    device=1
fi
if [ -z $USE_OPTUNA ]; then
    USE_OPTUNA=0
fi
if [ $DEVICE = "XPU" ]; then
    ZE_AFFINITY_MASK=0 python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/clip_finetune/${CFG}_prompt.yaml \
    --output-dir output/${TRAINER}_${PTL}_${DEEP}/${CFG}/${DATASET} \
    --xpu $device \
    --seed 123 \
    --use-optuna $USE_OPTUNA \
    TRAINER.COOP.ACC $ACCEPT \
    TRAINER.COOP.N_PLN $PTL \
    TRAINER.COOP.PMT_DEEP $DEEP \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC True \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS 0
else
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/clip_finetune/${CFG}_prompt.yaml \
    --output-dir output/${TRAINER}_${PTL}_${DEEP}/${CFG}/${DATASET} \
    --xpu $device \
    --seed 123 \
    --use-optuna $USE_OPTUNA \
    TRAINER.COOP.ACC $ACCEPT \
    TRAINER.COOP.N_PLN $PTL \
    TRAINER.COOP.PMT_DEEP $DEEP \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC True \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS 0
fi
