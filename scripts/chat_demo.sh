CUDA_VISIBLE_DEVICES=4 python src/cli_demo.py \
    --model_name_or_path /home/mediatek/models/Llama-2-13b-chat-hf/ \
    --template llama2 \
    --finetuning_type lora \
    --checkpoint_dir outputs/llama_chat_sft/checkpoint-1000
    