# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

feature_dir=clip_feat

for DATASET in OxfordPets
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 3
done
