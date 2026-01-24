
import torch
import logging
from PIL import Image
import gc # 引入垃圾回收


# @torch.no_grad()
# def rank_text2image_confidence(model, processor, caption, gallery_full_paths, max_batch_size=50, log_prefix=""):
#     """
#     计算 Query 与 Gallery 图片的匹配度 Logits。
#     """
#     device = model.device
#     all_scores = []
#     # 引导前缀，与训练时的输出格式一致
#     prefix_str = "The best matches the following description is image "

#     # 1. 分批推理 (防止 OOM)
#     for i in range(0, len(gallery_full_paths), max_batch_size):
#         batch_paths = gallery_full_paths[i : i + max_batch_size]
#         n_current = len(batch_paths)
#         images = [Image.open(p).convert("RGB") for p in batch_paths]
#         # print('n_current :', n_current)
#         # 构造 Prompt
#         messages = [{
#             "role": "user",
#             "content": [{"type": "image"} for _ in range(n_current)] + 
#                        [{"type": "text", "text": f"Select the image that best matches the following description of the pedestrian's appearance:\n\"{caption}\""}]
#         }]

#         # import pdb; pdb.set_trace()
#         # 应用 Chat 模板并手动拼接前缀
#         prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         full_prompt = prompt + prefix_str
        
#         inputs = processor(text=[full_prompt], images=[images], return_tensors="pt").to(device)
#         outputs = model(**inputs)
        
#         # 获取最后一个 Token 的 Logits
#         logits = outputs.logits[:, -1, :] 

#         # 提取 '1' 到 'n' 的 Logits
#         for j in range(1, n_current + 1):
#             token_id = processor.tokenizer.encode(str(j), add_special_tokens=False)[-1]
#             score = logits[0, token_id].item()
#             all_scores.append(score)
     
#     # 这里分批次处理了 为什么还是会outofmerry呢？

#     # 2. 全局排序
#     # ranked_indices 存储的是 gallery_full_paths 的下标
#     ranked_indices = sorted(range(len(all_scores)), key=lambda k: all_scores[k], reverse=True)
    
#     # 3. 日志记录：模型到底选了哪张图？
#     best_global_idx = ranked_indices[0]
#     # 计算该图在它所属 batch 中的相对序号 (例如 image 3)
#     # 注意：这里的 batch_idx 只是为了日志展示，模拟模型输出 "image X"
#     best_batch_idx = (best_global_idx % max_batch_size) + 1
    
#     logging.info(f"[{log_prefix}] Model Answer: \"{prefix_str}{best_batch_idx}\" (Global Index: {best_global_idx})")
    
#     return ranked_indices, all_scores

"""
爆显存后重新构造如下：

"""


@torch.no_grad()
def rank_text2image_confidence(model, processor, caption, gallery_full_paths, max_batch_size=5, log_prefix=""):
    device = model.device
    all_scores = []
    prefix_str = "The best matches the following description is image "

    for i in range(0, len(gallery_full_paths), max_batch_size):
        batch_paths = gallery_full_paths[i : i + max_batch_size]
        n_current = len(batch_paths)
        
        # 优化点 1: 这里可以考虑对过大的图进行 resize，防止单张图 token 爆炸
        # 但为了保持精度，我们先不动 resize，主要靠清理显存
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        
        messages = [{
            "role": "user",
            "content": [{"type": "image"} for _ in range(n_current)] + 
                       [{"type": "text", "text": f"Select the image that best matches the following description of the pedestrian's appearance:\n\"{caption}\""}]
        }]
        
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_prompt = prompt + prefix_str
        
        inputs = processor(text=[full_prompt], images=[images], return_tensors="pt").to(device)
        
        # 优化点 2: use_cache=False
        # 我们只需要 logits，不需要 past_key_values (KV Cache)，这能节省约 30%-50% 的推理显存
        outputs = model(**inputs, use_cache=False)
        
        logits = outputs.logits[:, -1, :] 

        for j in range(1, n_current + 1):
            token_id = processor.tokenizer.encode(str(j), add_special_tokens=False)[-1]
            all_scores.append(logits[0, token_id].item())
            
        # 优化点 3: 手动清理显存
        # 这一点在循环推理中非常重要，确保上一轮的计算图彻底释放
        del inputs, outputs, logits, images
        torch.cuda.empty_cache()
        gc.collect()

    ranked_indices = sorted(range(len(all_scores)), key=lambda k: all_scores[k], reverse=True)
    
    best_global_idx = ranked_indices[0]
    best_batch_idx = (best_global_idx % max_batch_size) + 1
    
    logging.info(f"[{log_prefix}] Model Answer: \"{prefix_str}{best_batch_idx}\" (Global Index: {best_global_idx})")
    
    return ranked_indices, all_scores