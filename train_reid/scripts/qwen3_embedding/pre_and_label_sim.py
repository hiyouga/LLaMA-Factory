import json
import os
import concurrent.futures
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_model():
    """
    加载Qwen3-Embedding-4B模型
    
    Returns:
        SentenceTransformer: 加载好的模型
    """
    model_path = '/home/wangrui/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-4B'
    print(f"正在加载模型: {model_path}")
    model = SentenceTransformer(model_path)
    print("模型加载完成")
    return model


def calculate_similarity(pair, model):
    """
    计算一对文本的相似度
    
    Args:
        pair: 包含index、predict、label和images_path的字典
        model: SentenceTransformer模型
    
    Returns:
        dict: 包含原始数据和相似度的字典
    """
    try:
        predict = pair['predict']
        label = pair['label']
        images_path = pair.get('images_path', '')
        
        # 编码文本，使用不同的prompt
        predict_embedding = model.encode([predict], prompt_name="query")
        label_embedding = model.encode([label], prompt_name="query")
        
        # 计算余弦相似度
        similarity = model.similarity(predict_embedding, label_embedding).item()
        
        return {
            'predict': predict,
            'label': label,
            'images_path': images_path,
            'sim_score': similarity
        }
    except Exception as e:
        print(f"计算相似度时出错 (索引: {pair.get('index', '未知')}): {e}")
        return {
            'predict': pair.get('predict', ''),
            'label': pair.get('label', ''),
            'images_path': pair.get('images_path', ''),
            'sim_score': 0.0
        }


def read_input_file(file_path):
    """
    读取输入文件
    
    Args:
        file_path: 输入文件路径
    
    Returns:
        list: 包含index、predict、label和images_path的字典列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")
    
    print(f"正在读取输入文件: {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, desc="读取数据")):
            try:
                item = json.loads(line.strip())
                if 'predict' in item and 'label' in item:
                    data.append({
                        'index': idx,
                        'predict': item['predict'],
                        'label': item['label'],
                        'images_path': item.get('images_path', '')  # 获取images_path，如果不存在则使用空字符串
                    })
            except json.JSONDecodeError as e:
                print(f"警告: 第{idx+1}行JSON解析错误: {e}")
    
    print(f"成功读取 {len(data)} 条数据")
    return data


def save_results(results, output_path, avg_score):
    """
    保存结果到文件
    
    Args:
        results: 相似度计算结果列表
        output_path: 输出文件路径
        avg_score: 平均相似度
    """
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"正在保存结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        # 保存每一行的结果
        for result in tqdm(results, desc="保存结果"):
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # 保存最后一行的平均相似度
        avg_result = {'average_sim_score': avg_score}
        f.write(json.dumps(avg_result, ensure_ascii=False) + '\n')
    
    print(f"结果保存完成，平均相似度: {avg_score:.4f}")


def main():
    """
    主函数
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='计算predict和label字段的相似度')
    
    # 添加输入文件路径参数
    parser.add_argument('--input_path', 
                        type=str, 
                        default='/home/wangrui/code/LLaMA-Factory/train_reid/data/local_data/infer_res/generated_predictions_with_images_sorted.jsonl',
                        help='输入文件路径，包含predict和label字段的JSONL文件')
    
    # 添加输出文件路径参数
    parser.add_argument('--output_path',
                        type=str,
                        default='/home/wangrui/code/LLaMA-Factory/train_reid/data/local_data/infer_res/prediction_sim_score.jsonl',
                        help='输出文件路径，保存相似度计算结果')
    
    # 解析参数
    args = parser.parse_args()
    
    # 获取输入输出路径
    input_path = args.input_path
    output_path = args.output_path
    
    print(f"输入文件路径: {input_path}")
    print(f"输出文件路径: {output_path}")
    
    # 加载模型
    model = load_model()
    
    # 读取输入数据
    data = read_input_file(input_path)
    
    # 并发计算相似度
    print("开始并发计算相似度...")
    results = []
    similarities = []
    
    # 设置并发数
    max_workers = min(16, len(data) if len(data) < 100 else 32)
    print(f"使用 {max_workers} 个线程进行并发计算")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_pair = {executor.submit(calculate_similarity, pair, model): pair for pair in data}
        
        # 处理结果
        for future in tqdm(concurrent.futures.as_completed(future_to_pair), total=len(data), desc="计算相似度"):
            result = future.result()
            results.append(result)
            similarities.append(result['sim_score'])
    
    # 计算平均相似度
    if similarities:
        avg_score = sum(similarities) / len(similarities)
        max_sim_score = max(similarities)
        min_sim_score = min(similarities)
        print(f"所有样本相似度计算完成，平均相似度: {avg_score:.4f}，最大相似度: {max_sim_score:.4f}，最小相似度: {min_sim_score:.4f}")
    else:
        avg_score = 0.0
        print("没有有效样本进行相似度计算")
    
    # 保存结果
    save_results(results, output_path, avg_score)


if __name__ == "__main__":
    main()
    #  python /home/wangrui/code/LLaMA-Factory/train_reid/scripts/qwen3_embedding/pre_and_label_sim.py --input_path /home/wangrui/code/LLaMA-Factory/train_reid/data/local_data/infer_res/generated_predictions_with_images_sorted.jsonl --output_path /home/wangrui/code/LLaMA-Factory/train_reid/data/local_data/infer_res/prediction_sim_score.jsonl