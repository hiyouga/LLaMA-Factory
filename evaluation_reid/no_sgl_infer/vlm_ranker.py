import torch
import logging
from PIL import Image

@torch.no_grad()
def rank_text2image_confidence(model, processor, caption, gallery_full_paths, max_batch_size=50, log_prefix=""):
    """
    计算 Query 与 Gallery 图片的匹配度 Logits。
    """
    device = model.device
    all_scores = []
    # 引导前缀，与训练时的输出格式一致
    prefix_str = "The best matches the following description is image "

    # 1. 分批推理 (防止 OOM)
    for i in range(0, len(gallery_full_paths), max_batch_size):
        batch_paths = gallery_full_paths[i : i + max_batch_size]
        n_current = len(batch_paths)
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        # print('n_current :', n_current)
        # 构造 Prompt
        messages = [{
            "role": "user",
            "content": [{"type": "image"} for _ in range(n_current)] + 
                       [{"type": "text", "text": f"Select the image that best matches the following description of the pedestrian's appearance:\n\"{caption}\""}]
        }]

        # import pdb; pdb.set_trace()
        # 应用 Chat 模板并手动拼接前缀
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_prompt = prompt + prefix_str
        
        inputs = processor(text=[full_prompt], images=[images], return_tensors="pt").to(device)
        outputs = model(**inputs)
        
        # 获取最后一个 Token 的 Logits
        logits = outputs.logits[:, -1, :] 

        # 提取 '1' 到 'n' 的 Logits
        for j in range(1, n_current + 1):
            token_id = processor.tokenizer.encode(str(j), add_special_tokens=False)[-1]
            score = logits[0, token_id].item()
            all_scores.append(score)

    # 2. 全局排序
    # ranked_indices 存储的是 gallery_full_paths 的下标
    ranked_indices = sorted(range(len(all_scores)), key=lambda k: all_scores[k], reverse=True)
    
    # 3. 日志记录：模型到底选了哪张图？
    best_global_idx = ranked_indices[0]
    # 计算该图在它所属 batch 中的相对序号 (例如 image 3)
    # 注意：这里的 batch_idx 只是为了日志展示，模拟模型输出 "image X"
    best_batch_idx = (best_global_idx % max_batch_size) + 1
    
    logging.info(f"[{log_prefix}] Model Answer: \"{prefix_str}{best_batch_idx}\" (Global Index: {best_global_idx})")
    
    return ranked_indices, all_scores