#!/bin/bash
# 建议使用 魔搭社区 下载资源挺快的，
export USE_MODELSCOPE_HUB=1
# 目前好像只能用一个显卡，暂时先这样，后续再考虑
export CUDA_VISIBLE_DEVICES=0
    # check requirement
python src/check_requirement.py
python src/train_web.py > llama-factory.log 2>&1

if [ $? -eq 0 ]; then
    echo "The llama factory webUI server has been started successfully,"
    echo "Access the address at http://0.0.0.0:7861"
else
    echo "start webUI failed, Please check the log llama-factory.log"
fi
echo "The log path is $(pwd)/llama-factory.log"
