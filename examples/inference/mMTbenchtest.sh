CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../../src/evaluate.py \
    --model_name_or_path /mnt/data/shesj/Trained/CommonAlign/DPO/QwenDPOSFT \
    --temperature 0.5 \
    --max_new_tokens 2048 \
    --template qwen \
    --finetuning_type full \
    --task mMTbench \
    --n_shot 0 \
    --n_iter 10 \
    --save_dir /mnt/data/shesj/EvalOut/mMTbench

