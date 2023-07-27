# python src/train_bash.py \
#     --stage ppo \
#     --model_name_or_path /home/mediatek/models/incite-7b-zh-base \
#     --use_fast_tokenizer true \
#     --lora_target query_key_value \
#     --prompt_template mr_chat \
#     --do_train \
#     --dataset alpaca_gpt4_zhtw \
#     --finetuning_type lora \
#     --checkpoint_dir ./outputs/incite-7b-zh-chat \
#     --reward_model ./outputs/incite-7b-zh-rm \
#     --output_dir ./outputs/incite-7b-zh-ppo \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate 1e-5 \
#     --num_train_epochs 1.0 \
#     --resume_lora_training False \
#     --plot_loss



accelerate launch --config_file accelerate_config.yaml src/train_bash.py \
    --stage ppo \
    --model_name_or_path /home/mediatek/models/incite-7b-zh-base \
    --use_fast_tokenizer true \
    --lora_target query_key_value \
    --prompt_template mr_chat \
    --do_train \
    --dataset alpaca_gpt4_zhtw \
    --finetuning_type lora \
    --checkpoint_dir ./outputs/incite-7b-zh-chat \
    --reward_model ./outputs/incite-7b-zh-rm \
    --output_dir ./outputs/incite-7b-zh-ppo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --resume_lora_training False \
    --plot_loss