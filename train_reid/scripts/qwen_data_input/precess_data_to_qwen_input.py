#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将数据转换为qwen输入格式
"""
import json
import os
from prompt import prompt, prompt_with_score, prompt_with_score_and_res

# 配置文件路径
INPUT_FILE = '/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_train.json'
OUTPUT_FILE = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_train_18k.json'

INPUT_TEST_FILE= '/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_test.json'
OUTPUT_TEST_FILE = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_test_1k.json'

IMAGE_DIR = '/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs'

# 新增配置：处理带分数的数据
INPUT_SCORE_FILE = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_caption_all_qwen.json"
OUTPUT_TRAIN_FILE_SCORE_1 = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_train_18k_retpreid_score.json'
OUTPUT_TEST_FILE_SCORE_1 = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_test_1k_retpreid_score.json'

OUTPUT_TRAIN_FILE_SCORE_3 = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_train_54k_retpreid_score_3.json'
OUTPUT_TEST_FILE_SCORE_3 = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_test_3k_retpreid_score_3.json'

OUTPUT_TRAIN_FILE_SCORE_3_WITH_RES = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_train_54k_retpreid_score_3_with_res_cot.json'
OUTPUT_TEST_FILE_SCORE_3_WITH_RES = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_test_3k_retpreid_score_3_with_res_cot.json'

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

# 处理带分数的数据，只使用captions最后一条和match_score最后一条
def process_score_data_last_one(input_file, output_train_file, output_test_file):
    """
    处理带分数的数据，只使用captions最后一条和match_score最后一条
    根据split拆分train和test
    
    Args:
        input_file: 输入文件路径
        output_train_file: 输出训练文件路径
        output_test_file: 输出测试文件路径
    """
    print(f"开始处理输入文件: {input_file}")
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取{len(data)}条数据")
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        return
    
    # 拆分训练集和测试集
    train_data = []
    test_data = []
    
    for idx, item in enumerate(data):
        try:
            split = item.get('split', 'train')
            captions = item.get('captions', [])
            match_scores = item.get('match_score', [])
            img_path = item.get('img_path', '')
            full_img_path = os.path.join(IMAGE_DIR, img_path)
            
            # 只处理有caption和match_score的数据
            if captions and match_scores:
                # 只使用最后一条caption和最后一条match_score
                last_caption = captions[-1]
                last_score = match_scores[-1]
                
                # 构建新的描述
                new_description = f"The final description of the image is: {last_caption}. In summary, the degree of relevance to the image is: {last_score}."
                
                # 构建qwen格式的数据
                qwen_item = {
                    "messages": [
                        {
                            "content": f"<image>{prompt_with_score}",
                            "role": "user"
                        },
                        {
                            "content": new_description,
                            "role": "assistant"
                        }
                    ],
                    "images": [
                        full_img_path
                    ]
                }
                
                # 根据split添加到对应数据集
                if split == 'train':
                    train_data.append(qwen_item)
                elif split == 'test':
                    test_data.append(qwen_item)
                
            # 显示进度
            if (idx + 1) % 1000 == 0:
                print(f"已处理{idx + 1}条数据")
                
        except Exception as e:
            print(f"处理第{idx}条数据失败: {e}")
            continue
    
    print(f"数据处理完成，训练集: {len(train_data)}条，测试集: {len(test_data)}条")
    
    # 保存训练集
    try:
        output_dir = os.path.dirname(output_train_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"训练集已保存到: {output_train_file}")
    except Exception as e:
        print(f"保存训练集失败: {e}")
    
    # 保存测试集
    try:
        output_dir = os.path.dirname(output_test_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print(f"测试集已保存到: {output_test_file}")
    except Exception as e:
        print(f"保存测试集失败: {e}")
    
    print("所有操作完成！")

# 处理带分数的数据，使用所有3个caption和对应的match_score[没有res]
def process_score_data_all_three(input_file, output_train_file, output_test_file):
    """
    处理带分数的数据，使用所有3个caption和对应的match_score
    根据split拆分train和test
    
    Args:
        input_file: 输入文件路径
        output_train_file: 输出训练文件路径
        output_test_file: 输出测试文件路径
    """
    print(f"开始处理输入文件: {input_file}")
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取{len(data)}条数据")
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        return
    
    # 拆分训练集和测试集
    train_data = []
    test_data = []
    
    for idx, item in enumerate(data):
        try:
            split = item.get('split', 'train')
            captions = item.get('captions', [])
            match_scores = item.get('match_score', [])
            img_path = item.get('img_path', '')
            full_img_path = os.path.join(IMAGE_DIR, img_path)
            
            # 确保caption和match_score数量一致
            if len(captions) == len(match_scores):
                # 遍历所有caption和对应的match_score
                for caption, score in zip(captions, match_scores):
                    # 构建新的描述
                    new_description = f"The final description of the image is:{caption}. In summary, the degree of relevance to the image is: {score}."
                    
                    # 构建qwen格式的数据
                    qwen_item = {
                        "messages": [
                            {
                                "content": f"<image>{prompt_with_score}",
                                "role": "user"
                            },
                            {
                                "content": new_description,
                                "role": "assistant"
                            }
                        ],
                        "images": [
                            full_img_path
                        ]
                    }
                    
                    # 根据split添加到对应数据集
                    if split == 'train':
                        train_data.append(qwen_item)
                    elif split == 'test':
                        test_data.append(qwen_item)
                
            # 显示进度
            if (idx + 1) % 1000 == 0:
                print(f"已处理{idx + 1}条数据")
                
        except Exception as e:
            print(f"处理第{idx}条数据失败: {e}")
            continue
    
    print(f"数据处理完成，训练集: {len(train_data)}条，测试集: {len(test_data)}条")
    
    # 保存训练集
    try:
        output_dir = os.path.dirname(output_train_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"训练集已保存到: {output_train_file}")
    except Exception as e:
        print(f"保存训练集失败: {e}")
    
    # 保存测试集
    try:
        output_dir = os.path.dirname(output_test_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print(f"测试集已保存到: {output_test_file}")
    except Exception as e:
        print(f"保存测试集失败: {e}")
    
    print("所有操作完成！")

# 处理带分数的数据，使用所有3个caption、match_score和reason
def process_score_data_with_reason(input_file, output_train_file, output_test_file):
    """
    处理带分数的数据，使用所有3个caption、match_score和reason
    根据split拆分train和test
    
    Args:
        input_file: 输入文件路径
        output_train_file: 输出训练文件路径
        output_test_file: 输出测试文件路径
    """
    print(f"开始处理输入文件: {input_file}")
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取{len(data)}条数据")
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        return
    
    # 拆分训练集和测试集
    train_data = []
    test_data = []
    
    for idx, item in enumerate(data):
        try:
            split = item.get('split', 'train')
            captions = item.get('captions', [])
            match_scores = item.get('match_score', [])
            reasons = item.get('reason', [])
            img_path = item.get('img_path', '')
            full_img_path = os.path.join(IMAGE_DIR, img_path)
            
            # 确保caption、match_score和reason数量一致
            if len(captions) == len(match_scores) and len(captions) == len(reasons):
                # 遍历所有caption、match_score和reason
                for caption, score, reason in zip(captions, match_scores, reasons):
                    # 构建新的描述，确保语言通顺
                    new_description = f"The final description of the image is: {caption}. The reason is {reason}. Therefore, the degree of relevance to the image is: {score}."
                    
                    # 构建qwen格式的数据
                    qwen_item = {
                        "messages": [
                            {
                                "content": f"<image>{prompt_with_score_and_res}",
                                "role": "user"
                            },
                            {
                                "content": new_description,
                                "role": "assistant"
                            }
                        ],
                        "images": [
                            full_img_path
                        ]
                    }
                    
                    # 根据split添加到对应数据集
                    if split == 'train':
                        train_data.append(qwen_item)
                    elif split == 'test':
                        test_data.append(qwen_item)
                
            # 显示进度
            if (idx + 1) % 1000 == 0:
                print(f"已处理{idx + 1}条数据")
                
        except Exception as e:
            print(f"处理第{idx}条数据失败: {e}")
            continue
    
    print(f"数据处理完成，训练集: {len(train_data)}条，测试集: {len(test_data)}条")
    
    # 保存训练集
    try:
        output_dir = os.path.dirname(output_train_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"训练集已保存到: {output_train_file}")
    except Exception as e:
        print(f"保存训练集失败: {e}")
    
    # 保存测试集
    try:
        output_dir = os.path.dirname(output_test_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print(f"测试集已保存到: {output_test_file}")
    except Exception as e:
        print(f"保存测试集失败: {e}")
    
    print("所有操作完成！")


if __name__ == "__main__":
    # 调用原有函数
    # convert_to_qwen_format(INPUT_FILE, OUTPUT_FILE)
    # convert_to_qwen_format(INPUT_TEST_FILE, OUTPUT_TEST_FILE)
    
    # 调用新增函数
    # 处理最后一条caption和score
    process_score_data_last_one(INPUT_SCORE_FILE, OUTPUT_TRAIN_FILE_SCORE_1, OUTPUT_TEST_FILE_SCORE_1)
    
    # 处理所有3个caption和score
    process_score_data_all_three(INPUT_SCORE_FILE, OUTPUT_TRAIN_FILE_SCORE_3, OUTPUT_TEST_FILE_SCORE_3)
    
    # 处理所有3个caption、score和reason
    process_score_data_with_reason(INPUT_SCORE_FILE, OUTPUT_TRAIN_FILE_SCORE_3_WITH_RES, OUTPUT_TEST_FILE_SCORE_3_WITH_RES)
