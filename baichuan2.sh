
cd LLaMA-Factory

export CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --standalone --nnodes=1 --nproc-per-node=4 src/train_bash.py \
    --stage sft \
    --model_name_or_path ../pretrain/Baichuan2-7B-Chat \
    --do_train \
    --dataset sft_1028 \
    --template baichuan2 \
    --finetuning_type lora \
    --lora_target W_pack \
    --output_dir ../finetuned/baichuan2-7B-chat-sft_10282 \
    --overwrite_cache \
    --cutoff_len 2100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_steps 200 \
    --save_total_limit 20 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --learning_rate 5e-5 \
    --num_train_epochs 8.0 \
    --plot_loss \
    --fp16 \
    --overwrite_output_dir \
    --flash_attn \
    # --memory_efficient_attention
    # --quantization_bit 8
    # --do_eval \
    # --split all \
    # --val_size 0.01 \
    # --evaluation_strategy steps \
    # --eval_steps 10 \
    
