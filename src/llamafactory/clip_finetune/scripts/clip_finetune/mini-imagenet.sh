#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

train_directory_path="/home/wangjilong/data/mini-imagenet/train/"
val_directory_path="/home/wangjilong/data/mini-imagenet/val/"
test_directory_path="/home/wangjilong/data/mini-imagenet/test/"
if [ -d $val_directory_path ];then
    rm -rf $val_directory_path
fi
if [ -d $test_directory_path ];then
    rm -rf $test_directory_path
fi
mkdir -p $val_directory_path
mkdir -p $test_directory_path
for file in $(ls $train_directory_path)
do
    mkdir -p $val_directory_path/$file
    files=$(ls "$train_directory_path/$file")
    random_files=$(echo "$files" | tr ' ' '\n' | shuf -n 50)
    for random_file in $random_files
    do
        mv $train_directory_path/$file/$random_file $val_directory_path/$file/
    done
done
for file in $(ls $train_directory_path)
do
    mkdir -p $test_directory_path/$file
    files=$(ls "$train_directory_path/$file")
    random_files=$(echo "$files" | tr ' ' '\n' | shuf -n 100)
    for random_file in $random_files
    do
        mv $train_directory_path/$file/$random_file $test_directory_path/$file/
    done
done
