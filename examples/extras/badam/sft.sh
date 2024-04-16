# BAdam layer-wise
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ../../../src/train_bash.py \
--stage sft \
--do_train \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--dataset alpaca_gpt4_en,glaive_toolcall \
--dataset_dir ../../../data \
--template default \
--finetuning_type full \
--output_dir ../../../saves/LLaMA2-7B/badam \
--overwrite_cache \
--overwrite_output_dir \
--cutoff_len 1024 \
--preprocessing_num_workers 32 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 5 \
--gradient_accumulation_steps 2 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--warmup_steps 20 \
--save_steps 100 \
--eval_steps 100 \
--evaluation_strategy steps \
--load_best_model_at_end \
--learning_rate 5e-5 \
--num_train_epochs 3.0 \
--val_size 0.1 \
--plot_loss \
--use_badam \
--switch_mode descending \
--badam_verbose 2 \
--switch_block_every 50

