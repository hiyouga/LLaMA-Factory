1.镜像环境准备
docker run \
  -p 8888:8888 \
  -v /home/cwd/codes:/home/jovyan \
  -u 0 \
  --env GRANTSUDO="yes" \
  --gpus all \
  --rm \
  hub.wls.195803.xyz/library/base-notebook-cwd:11.8.0-devel-ubuntu22.04-python3.10-f8bca4e

2.数据集准备
data文件夹
2.1 修改LLaMA-Factory/data/dataset_info.json文件对应数据集名称
2.2 下载对应数据集json文件

3.下载模型文件
3.1 模型文件放在meta-llama文件
3.2 修改train、inference和merage三个文件
   LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml LLaMA-Factory/examples/inference/llama3_lora_sft.yaml LLaMA-Factory/examples/merge_lora/llama3_lora_sft.yaml