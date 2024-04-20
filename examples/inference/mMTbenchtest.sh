model_name=$1
model_path=$2

#bash mMTbenchtest.sh QwenSixDPO /mnt/data/shesj/Trained/CommonAlign/DPO/QwenSixDPO
#bash mMTbenchtest.sh QwenSixSFTBase /mnt/data/shesj/Trained/RL4CoT/SFT/Qwen1_5_7b_Chat_SixLang
#bash mMTbenchtest.sh QwenDPOSFT /mnt/data/shesj/Trained/CommonAlign/DPO/QwenDPOSFT
#bash mMTbenchtest.sh Qwen /mnt/data/shesj/PLM/Qwen1.5-7B-Chat
#bash mMTbenchtest.sh Qwen /mnt/data/shesj/PLM/Qwen1.5-7B-Chat

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../../src/evaluate.py \
    --model_name $model_name \
    --model_name_or_path $model_path \
    --temperature 0.3 \
    --max_new_tokens 2048 \
    --template qwen \
    --finetuning_type full \
    --task mMTbench \
    --n_shot 0 \
    --n_iter 10 \
    --save_dir /mnt/data/shesj/EvalOut/mMTbench


