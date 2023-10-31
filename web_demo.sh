
# chatglm3
export CUDA_VISIBLE_DEVICES="0,1"
python src/web_demo.py \
    --model_name_or_path ../pretrain/chatglm3-6b-base \
    --template chatglm3 \
    --finetuning_type lora \
    --checkpoint_dir ../finetuned/chatglm3-base-sft_1028/checkpoint-700


# baichuan2

# export CUDA_VISIBLE_DEVICES="0,1"
# python src/web_demo.py \
#     --model_name_or_path ../pretrain/Baichuan2-7B-Base\
#     --template baichuan2 \
#     --finetuning_type lora \
#     --checkpoint_dir ../finetuned/baichuan2-7B-base-sft_1028/checkpoint-1800
