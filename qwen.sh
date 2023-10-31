
# cd LLaMA-Factory

export CUDA_VISIBLE_DEVICES="1"
torchrun --standalone --nnodes=1 --nproc-per-node=1 src/train_bash.py \
    --stage sft \
    --model_name_or_path ../pretrain/Qwen-7B \
    --do_train \
    --dataset sft1030 \
    --template chatml \
    --finetuning_type lora \
    --lora_target c_attn \
    --output_dir ../finetuned/Qwen-7B-sft1030 \
    --cutoff_len 2100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 100 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --fp16 \
    --do_eval \
    --val_size 0.1 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --flash_attn \
    --checkpoint_dir ../finetuned/Qwen-7B-sft1030/checkpoint-500 \
    # --memory_efficient_attention
    # --quantization_bit 8
    # --split all \--overwrite_output_dir \--overwrite_cache \
    
    
