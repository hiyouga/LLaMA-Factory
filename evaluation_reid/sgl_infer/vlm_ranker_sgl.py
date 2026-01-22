import requests
import json

def rank_text2image_sglang(caption, image_full_paths, endpoint="http://localhost:30000"):
    """
    通过 SGLang Runtime 发送请求。
    利用 SGLang 的 select 功能直接获取各选项的 Logits。
    """
    # 构造 Prompt：与训练格式完全一致
    # 注意：Qwen2.5-VL 在 SGLang 中通常使用 [IMAGE] 占位符或特定 API
    
    # 构造数字候选项字符串
    choices = [str(i+1) for i in range(len(image_full_paths))]
    
    # 这里的 prompt 结构适配您的训练数据
    prompt = f"Select the image that best matches the following description of the pedestrian's appearance:\n\"{caption}\"\nThe best matches the following description is image "
    
    # SGLang 请求体
    data = {
        "text": prompt,
        "images": image_full_paths,
        "sampling_params": {
            "max_new_tokens": 1,
            "stop": [" "],
        },
        # 使用 select 功能获取特定 token 的 logprobs
        "choices": choices 
    }
    
    response = requests.post(f"{endpoint}/generate", json=data)
    res_json = response.json()
    
    # 获取每个选项的 logprob 转化为得分
    # 注意：实际 API 字段需参考 SGLang 最新文档，通常在 'choices_logprobs'
    scores = res_json.get("choices_logprobs", [0.0] * len(choices))
    
    # 排序
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked_indices, scores