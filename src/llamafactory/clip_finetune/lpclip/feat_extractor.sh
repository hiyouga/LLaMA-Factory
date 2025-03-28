# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# sh feat_extractor.sh
DATA=/path/to/datasets
OUTPUT='./clip_feat/'
SEED=1

# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
for DATASET in oxford_pets
do
    for SPLIT in train val test
    do
        python feat_extractor.py \
        --split ${SPLIT} \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file ../configs/datasets/${DATASET}.yaml \
        --config-file ../configs/trainers/CoOp/rn50_val.yaml \
        --output-dir ${OUTPUT} \
        --eval-only
    done
done
