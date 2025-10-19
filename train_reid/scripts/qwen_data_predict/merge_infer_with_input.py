import json
import os
import argparse
from tqdm import tqdm
import re


def extract_image_prefix(file_path):
    """
    从图像路径中提取文件名前缀（如从3901_c14_0007.jpg中提取3901）
    
    Args:
        file_path: 图像路径
    
    Returns:
        提取的数字前缀，如果无法提取则返回0
    """
    try:
        # 获取文件名
        if isinstance(file_path, list) and file_path:
            file_name = os.path.basename(file_path[0])
        else:
            file_name = os.path.basename(str(file_path))
        
        # 使用正则表达式提取数字前缀
        match = re.match(r'(\d+)', file_name)
        if match:
            return int(match.group(1))
        return 0
    except Exception:
        return 0


def sort_by_image_prefix(input_file, output_file):
    """
    按照images_path中的文件名前缀（如3901）进行升序排序
    
    Args:
        input_file: 输入文件路径 (generated_predictions_with_images.jsonl)
        output_file: 排序后的输出文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取数据
    print(f"正在读取数据: {input_file}")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, desc="读取数据")):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"警告: 第{idx+1}行JSON解析错误: {e}")
    
    print(f"成功读取 {len(data)} 条记录")
    
    # 排序
    print("正在按照图像前缀排序...")
    sorted_data = sorted(data, key=lambda x: extract_image_prefix(x.get('images_path', [])))
    
    # 保存排序后的结果
    print(f"正在保存排序结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(sorted_data, desc="保存结果"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"排序完成! 共处理 {len(sorted_data)} 条记录")


def merge_infer_with_input(predictions_path, input_data_path, output_path):
    """
    合并预测结果和输入数据
    
    Args:
        predictions_path: 预测结果文件路径 (generated_predictions.jsonl)
        input_data_path: 输入数据文件路径 (mllm_reid_test_1k.json)
        output_path: 输出文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"预测结果文件不存在: {predictions_path}")
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"输入数据文件不存在: {input_data_path}")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取输入数据，提取images字段
    print(f"正在读取输入数据: {input_data_path}")
    with open(input_data_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # 提取所有images路径
    images_paths = []
    for item in input_data:
        if 'images' in item:
            images_paths.append(item['images'])
    
    print(f"成功读取 {len(images_paths)} 条输入数据")
    
    # 读取预测结果并合并
    print(f"正在读取并合并预测结果: {predictions_path}")
    results = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, desc="处理预测结果")):
            try:
                pred_item = json.loads(line.strip())
                
                # 添加images_path字段
                if idx < len(images_paths):
                    pred_item['images_path'] = images_paths[idx]
                else:
                    print(f"警告: 预测结果行数({idx+1})超过输入数据行数({len(images_paths)})")
                    pred_item['images_path'] = None
                
                results.append(pred_item)
            except json.JSONDecodeError as e:
                print(f"警告: 第{idx+1}行JSON解析错误: {e}")
    
    # 保存结果
    print(f"正在保存合并结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"合并完成! 共处理 {len(results)} 条记录")


def main():
    """
    主函数
    """
    # 默认配置
    default_predictions = '/home/wangrui/code/LLaMA-Factory/saves/qwen2_5vl-7b/lora/predict_new1/generated_predictions.jsonl'
    default_input = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_test_1k.json'
    default_output = '/home/wangrui/code/LLaMA-Factory/train_reid/local_data/infer_res/generated_predictions_with_images.jsonl'
    default_sorted_output = '/home/wangrui/code/LLaMA-Factory/train_reid/local_data/infer_res/generated_predictions_with_images_sorted.jsonl'
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='合并预测结果和输入数据，并支持排序功能')
    parser.add_argument('--predictions', type=str, default=default_predictions,
                        help='预测结果文件路径 (generated_predictions.jsonl)')
    parser.add_argument('--input', type=str, default=default_input,
                        help='输入数据文件路径 (mllm_reid_test_1k.json)')
    parser.add_argument('--output', type=str, default=default_output,
                        help='合并后的输出文件路径')
    parser.add_argument('--sort', action='store_true',
                        help='是否对合并后的结果按图像前缀排序')
    parser.add_argument('--sorted-output', type=str, default=default_sorted_output,
                        help='排序后的输出文件路径')
    parser.add_argument('--sort-only', type=str,
                        help='仅执行排序操作，指定输入文件路径')
    
    args = parser.parse_args()
    
    try:
        # 如果指定了sort-only，则只执行排序
        if args.sort_only:
            sort_by_image_prefix(args.sort_only, args.sorted_output)
        else:
            # 先执行合并
            merge_infer_with_input(args.predictions, args.input, args.output)
            
            # 如果需要排序，则执行排序
            if args.sort:
                sort_by_image_prefix(args.output, args.sorted_output)
    except Exception as e:
        print(f"错误: {e}")
        exit(1)


if __name__ == "__main__":
    main()