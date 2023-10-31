cd LLaMA-Factory
export CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --standalone --nnodes=1 --nproc-per-node=4 src/train_bash.py \
    --stage sft \
    --model_name_or_path ../pretrain/chatglm3-6b-base \
    --do_train \
    --do_eval \
    --split all \
    --dataset sft_1028 \
    --template chatglm3 \
    --finetuning_type lora \
    --lora_target query_key_value \
    --output_dir ../finetuned/chatglm3-base-sft_1028 \
    --overwrite_cache \
    --cutoff_len 2100 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --val_size 0.01 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --learning_rate 5e-5 \
    --num_train_epochs 8.0 \
    --plot_loss \
    --fp16 \
    --overwrite_output_dir
    # --save_total_limit 20 \