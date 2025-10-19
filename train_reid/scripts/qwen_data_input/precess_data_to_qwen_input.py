#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将数据转换为qwen输入格式
"""
import json
import os
from prompt import prompt

# 配置文件路径
INPUT_FILE = '/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_train.json'
OUTPUT_FILE = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_train_18k.json'

INPUT_TEST_FILE= '/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_test.json'
OUTPUT_TEST_FILE = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_test_1k.json'

IMAGE_DIR = '/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs'

def convert_to_qwen_format(input_file, output_file):
    """
    将数据转换为qwen输入格式
    1. 读取data_captions_train_8k.json
    2. 转换为mllm_demo_test.json的格式
    3. 保存到指定位置
    """
    print(f"开始读取输入文件: {input_file}")
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取{len(data)}条数据")
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        return
    
    # 转换数据格式
    converted_data = []
    for idx, item in enumerate(data):
        try:
            # 构建图片路径
            img_path = item.get('img_path', '')
            full_img_path = os.path.join(IMAGE_DIR, img_path)
            
            # 提取caption_qwen中的描述
            caption_qwen = item.get('caption_qwen', '')
            if caption_qwen.startswith('result: '):
                description = caption_qwen[len('result: '):]
            else:
                description = caption_qwen
            
            # 构建qwen格式的数据
            qwen_item = {
                "messages": [
                    {
                        "content": f"<image>{prompt}",
                        "role": "user"
                    },
                    {
                        "content": description,
                        "role": "assistant"
                    }
                ],
                "images": [
                    full_img_path
                ]
            }
            
            converted_data.append(qwen_item)
            
            # 显示进度
            if (idx + 1) % 1000 == 0:
                print(f"已处理{idx + 1}条数据")
                
        except Exception as e:
            print(f"处理第{idx}条数据失败: {e}")
            continue
    
    print(f"数据转换完成，成功转换{len(converted_data)}条数据")
    
    # 保存结果
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")
        return
    
    print("所有操作完成！")

if __name__ == "__main__":
    # convert_to_qwen_format(INPUT_FILE, OUTPUT_FILE)
    convert_to_qwen_format(INPUT_TEST_FILE, OUTPUT_TEST_FILE)