#!/bin/bash
# 请确保安装了 sglang: pip install sglang
# --mem-fraction-static 0.8 留出部分显存给系统
# --model-path 填入你的 Qwen2.5-VL 路径

python -m sglang.launch_server \
    --model-path /home/wangrui/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct \
    --port 30000 \
    --tokenizer-path /home/wangrui/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct \
    --mem-fraction-static 0.8 \
    --chat-template qwen2-vl