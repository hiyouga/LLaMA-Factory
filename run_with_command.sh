export USE_MODELSCOPE_HUB=1 # 使用 modelscope 下载模型 
export MODELSCOPE_CACHE="/workspace/LLaMA-Factory/modelscope"
export MODELSCOPE_MODULES_CACHE="/workspace/LLaMA-Factory/modelscope/modules"

llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
