
export CUDA_VISIBLE_DEVICES="0,1"
python src/cli_demo.py \
    --model_name_or_path ../pretrain/Baichuan2-7B-Base\
    --template baichuan2 \
    --finetuning_type lora \
    --checkpoint_dir ../finetuned/baichuan2-7B-base-sft_1028/checkpoint-1800