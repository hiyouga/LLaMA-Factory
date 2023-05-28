# LLaMA Efficient Tuning

1. Download the weights of the LLaMA models.
2. Convert them to HF format using this [script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)

```python
python convert_llama_weights_to_hf.py \
    --input_dir path_to_llama_weights --model_size 7B --output_dir llama_7b
```

3. Fine-tune the LLaMA models.

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
    --model_name_or_path llama_7b \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --finetuning_type lora \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --fp16
```
