#!/bin/bash

# Exit script on error
set -e

# bash train_template.sh 2,3 10086 /home/data/models/Llama-2-7b-hf alpaca_en default 2e-5 8 128 3
# bash train_template.sh 6,7 10088 /home/data/models/Llama-2-7b-hf alpaca_en default 2e-5 16 32 3
# bash train_template.sh 4,5 10089 /home/data/models/Llama-2-7b-hf alpaca_en default 2e-5 8 64 3
# bash train_template.sh 4,5 10090 /home/data/models/Llama-2-7b-hf evol_instruct vicuna 2e-5 16 8 3
# bash train_template.sh 1,5,6,7 10092 /home/data/models/Llama-2-7b-hf evol_instruct_local vicuna 2e-5 16 8 3
# bash train_template.sh 4,5,6,7 10093 /home/data/models/Llama-2-7b-hf alpaca_en alpaca 2e-5 16 4 3

# bash train_template.sh 3 10094 /home/data/models/Llama-2-7b-hf evol_instruct vicuna 2e-5 8 4 3
# bash train_template.sh 3 10094 /home/data/models/Llama-2-7b-hf evol_instruct_local vicuna 2e-5 8 4 3
# bash train_template.sh 4,5,6,7 10095 /home/data/models/Llama-2-7b-hf alpaca_en default 2e-5 16 2 3
# bash train_template.sh 4,5 10104 /home/data/models/Llama-2-7b-hf alpaca_en_1% vicuna 2e-5 16 4 3
# bash train
# bash train_template.sh 0,1,2,3 10106 /home/data/models/Llama-2-7b-hf evol_instruct_10%_arrow vicuna 2e-5 16 2 3
# bash train_template.sh 0,1,2,3 10108 /home/data/models/Llama-2-7b-hf alpaca_en_1% default 2e-5 16 2 3

# llama2-chat 1%, 10%, 50%, 100% alpaca_en
# bash train_template.sh 4,5,6,7 10000 /home/data/models/Llama-2-7b-chat-hf alpaca_en_1% llama2 2e-5 16 2 3
# bash train_template.sh 4,5,6,7 10001 /home/data/models/Llama-2-7b-chat-hf alpaca_en_10% llama2 2e-5 16 2 3
# bash train_template.sh 4,5,6,7 10002 /home/data/models/Llama-2-7b-chat-hf alpaca_en_50% llama2 2e-5 16 2 3
# bash train_template.sh 4,5,6,7 10003 /home/data/models/Llama-2-7b-chat-hf alpaca_en llama2 2e-5 16 2 3

# evol_instruct local vs hf vs local json vs hf_formatted
# bash train_template.sh 0,1,2,3 10111 /home/data/models/Llama-2-7b-hf evol_instruct_local vicuna 2e-5 16 2 3
# bash train_template.sh 4,5,6,7 10112 /home/data/models/Llama-2-7b-hf evol_instruct vicuna 2e-5 16 2 3
# bash train_template.sh 4,5,6,7 10112 /home/data/models/Llama-2-7b-hf evol_instruct_hf vicuna 2e-5 16 2 3
# bash train_template.sh 0,1,2,3 10113 /home/data/models/Llama-2-7b-hf evol_instruct_json vicuna 2e-5 16 2 3

# test alpaca_100, evol_instruct_100 tokens
# bash train_template.sh 0,1,2,3 10114 /home/data/models/Llama-2-7b-hf alpaca_en_100 default 2e-5 16 2 3

# exinren
# bash train_template.sh 0,1,2,3,4,5,6,7 10001 /home/data/models/Llama-2-7b-hf evol_instruct_local vicuna 2e-5 8 2 3

# CW vs SFT, Mistral & Qwen
# bash train_template.sh 0,1,2,3 10114 /home/data/models/mistralai/Mistral-7B-Instruct-v0.2 alpaca_en_10 mistral 2e-5 16 2 3 &
# bash train_template.sh 4,5,6,7 10115 /home/data/models/mistralai/Mistral-7B-Instruct-v0.2 evol_instruct_10 mistral 2e-5 16 2 3
# bash train_template.sh 0,1,2,3 10116 /home/data/models/mistralai/Mistral-7B-Instruct-v0.2 alpaca_en_50 mistral 2e-5 16 2 3 &
# bash train_template.sh 4,5,6,7 10117 /home/data/models/mistralai/Mistral-7B-Instruct-v0.2 evol_instruct_50 mistral 2e-5 16 2 3
bash train_template.sh 0,1,2,3 10118 /home/data/models/mistralai/Mistral-7B-Instruct-v0.2 alpaca_en_100 mistral 2e-5 16 2 3
bash train_template.sh 4,5,6,7 10117 /home/data/models/mistralai/Mistral-7B-Instruct-v0.2 evol_instruct_100 mistral 2e-5 16 2 3