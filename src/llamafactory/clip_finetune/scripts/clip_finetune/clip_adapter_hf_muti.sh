#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# custom config

TRAINER=CLIP_Adapter_hf
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
ACCEPT=$3
DEVICE=$4
if [ -z $ACCEPT ]; then
    ACCEPT=0
fi
if [ -z $DEVICE ]; then
    DEVICE="cuda"
    device=0
fi
if [ $DEVICE = "XPU" ]; then
    device=1
fi
echo "adapter"
mpiexec -n 2 -l python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/clip_finetune/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--xpu $device \
TRAINER.COOP.ACC $ACCEPT \
TRAINER.COOP.N_CTX 16 \
TRAINER.COOP.CSC True \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0
# echo "VPT"
# TRAINER=CLIP_VPT_hf
# PTL=1
# DEEP=True
# mpirun -n 2 -l python train.py \
#     --root ${DATA} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/clip_finetune/${CFG}_prompt.yaml \
#     --output-dir output/${TRAINER}_${PTL}_${DEEP}/${CFG}/${DATASET} \
#     --xpu $device \
#     TRAINER.COOP.ACC $ACCEPT \
#     TRAINER.COOP.N_PLN $PTL \
#     TRAINER.COOP.PMT_DEEP $DEEP \
#     TRAINER.COOP.N_CTX 16 \
#     TRAINER.COOP.CSC True \
#     TRAINER.COOP.CLASS_TOKEN_POSITION end \
#     DATASET.NUM_SHOTS 0

# echo "bias"
# TRAINER=CLIP_Bias_hf
# mpiexec -n 2 -l python train.py \
#     --root ${DATA} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/clip_finetune/${CFG}_bias.yaml \
#     --output-dir output/${TRAINER}/${CFG}/${DATASET} \
#     --xpu $device \
#     TRAINER.COOP.ACC $ACCEPT \
#     TRAINER.COOP.N_CTX 16 \
#     TRAINER.COOP.CSC True \
#     TRAINER.COOP.CLASS_TOKEN_POSITION end \
#     DATASET.NUM_SHOTS 0

# echo "abs"
# TRAINER=CLIP_Fullfinetune_hf
# mpiexec -n 2 -l python train.py \
#     --root ${DATA} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/clip_finetune/${CFG}_ori_abs.yaml \
#     --output-dir output/${TRAINER}_abs_g1/${CFG}/${DATASET} \
#     --xpu $device \
#     TRAINER.COOP.ACC $ACCEPT \
#     TRAINER.COOP.N_CTX 16 \
#     TRAINER.COOP.CSC True \
#     TRAINER.COOP.CLASS_TOKEN_POSITION end \
#     DATASET.NUM_SHOTS 0

# echo "full"
# TRAINER=CLIP_Fullfinetune_hf
# mpirun -n 2 -l python train.py \
#     --root ${DATA} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/clip_finetune/${CFG}_ori.yaml \
#     --output-dir output/${TRAINER}/${CFG}/${DATASET} \
#     --xpu $device \
#     TRAINER.COOP.ACC $ACCEPT \
#     TRAINER.COOP.N_CTX 16 \
#     TRAINER.COOP.CSC True \
#     TRAINER.COOP.CLASS_TOKEN_POSITION end \
#     DATASET.NUM_SHOTS 0
