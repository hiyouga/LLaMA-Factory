"""
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  --stage sft \
  --model_name_or_path THUDM/chatglm3-6b \
  --do_train \
  --dataset_dir ./data \
  --dataset middle_tcm_sharegpt \
  --template chatglm3 \
  --finetuning_type lora \
  --output_dir ./checkpoints/chatglm3-tcm \
  --overwrite_cache \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_steps 500 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --fp16
"""