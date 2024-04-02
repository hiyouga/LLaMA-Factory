> [!WARNING]
> Merging LoRA weights into a quantized model is not supported.

> [!TIP]
> Use `--model_name_or_path path_to_model` solely to use the exported model or model fine-tuned in full/freeze mode.
>
> Use `CUDA_VISIBLE_DEVICES=0`, `--export_quantization_bit 4` and `--export_quantization_dataset data/c4_demo.json` to quantize the model with AutoGPTQ after merging the LoRA weights.


Usage:

- `merge.sh`: merge the lora weights
- `quantize.sh`: quantize the model with AutoGPTQ (must after merge.sh, optional)
