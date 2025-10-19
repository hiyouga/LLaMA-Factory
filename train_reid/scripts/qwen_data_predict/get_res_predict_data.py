#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从generated_predictions.jsonl文件中提取predict字段
"""
import json
import os
import argparse

def extract_predictions(input_file, output_file=None):
    """
    从JSONL文件中提取predict字段
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出文件路径，如果不指定则不保存
    
    Returns:
        提取的predictions列表
    """
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在 - {input_file}")
        return []
    
    predictions = []
    processed_count = 0
    error_count = 0
    
    print(f"开始从文件提取predict字段: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析JSON行
                    data = json.loads(line)
                    
                    # 提取predict字段
                    if 'predict' in data:
                        predictions.append(data['predict'])
                        processed_count += 1
                    else:
                        print(f"警告：第{line_number}行没有predict字段")
                        error_count += 1
                    
                    # 显示进度
                    if (line_number) % 1000 == 0:
                        print(f"已处理{line_number}行...")
                        
                except json.JSONDecodeError as e:
                    print(f"错误：第{line_number}行JSON解析失败: {e}")
                    error_count += 1
                    continue
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []
    
    print(f"提取完成！")
    print(f"成功处理: {processed_count}条")
    print(f"处理失败: {error_count}条")
    print(f"总共提取: {len(predictions)}个predict值")
    
    # 保存结果到文件
    if output_file:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 保存为JSON文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
    
    return predictions

def main():

    # 默认配置
    default_input = '/home/wangrui/code/LLaMA-Factory/saves/qwen2_5vl-7b/lora/predict_new1/generated_predictions.jsonl'
    default_output = '/home/wangrui/code/LLaMA-Factory/train_reid/local_data/infer_res/predictions_lora_bs1.json'
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='从JSONL文件中提取predict字段')
    parser.add_argument('--input', '-i', type=str, default=default_input,
                      help='输入的JSONL文件路径')
    parser.add_argument('--output', '-o', type=str, default=default_output,
                      help='输出的JSON文件路径')
    
    args = parser.parse_args()
    
    # 执行提取
    extract_predictions(args.input, args.output)

if __name__ == "__main__":
    main()