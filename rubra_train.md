rubra_train.md

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --finetuning_type lora \
    --template mistral \
    --dataset_dir data \
    --dataset glaive_toolcall \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --output_dir saves/Mistral-7B-v0.2-Chat/lora/train_2024-03-05-15-51-08 \
    --fp16 True \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --plot_loss True
```