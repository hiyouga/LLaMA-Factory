export DS_SKIP_CUDA_CHECK=1 
export DISABLE_VERSION_CHECK=1  # if necessary
# sft
deepspeed --hostfile=hostfile.2nodes src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path xxx \
    --dataset sft_toy \
    --template qwen \
    --finetuning_type full \
    --output_dir output/sft-test \
    --cache_dir .cache \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --learning_rate 5e-6 \
    --num_train_epochs 4 \
    --plot_loss \
    --save_only_model True \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
    --bf16 True \
    --flash_attn fa2 \
    --gradient_checkpointing True \
    --seed 42 \
    --sequence_parallel_size 8 \
    --packing True \
    --preprocessing_num_workers 32 \
    --report_to tensorboard

# dpo
deepspeed --hostfile=hostfile.1mac src/train.py \
    --stage dpo \
    --do_train \
    --model_name_or_path xxx \
    --dataset dpo_toy \
    --template qwen \
    --finetuning_type full \
    --pref_beta 0.1 \
    --pref_loss sigmoid \
    --output_dir output/debug \
    --cache_dir .cache \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --save_steps 2000 \
    --save_strategy steps \
    --learning_rate 1e-6 \
    --num_train_epochs 10 \
    --plot_loss \
    --save_only_model True \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
    --flash_attn fa2 \
    --gradient_checkpointing True \
    --bf16 True \
    --ddp_timeout 180000000 \
    --seed 42 \
    --sequence_parallel_size 4
