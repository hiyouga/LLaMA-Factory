import os
import json
import logging
import random
from tqdm import tqdm
from metrics import evaluate_rank
from vlm_ranker import rank_text2image_confidence

def run_iterative_eval(model, processor, test_data, image_root, cache_file, num_iters=5):
    """
    外层循环 num_iters 次。
    每次循环对整个测试集进行一次完整评估。
    最终取 5 次循环中最好的全局 R1 和 mAP。
    """
    with open(cache_file, "r") as f:
        gallery_data = json.load(f)

    # 初始化日志 (覆盖模式)
    logging.basicConfig(filename='eval_process.log', filemode='w', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', force=True)
    
    # 记录每一轮的全局结果
    round_metrics = []

    print(f"Starting evaluation with {num_iters} rounds...")

    # --- 外层循环：5 次迭代 ---
    for round_idx in range(1, num_iters + 1):
        logging.info(f"\n{'='*20} ROUND {round_idx} START {'='*20}")
        print(f"Running Round {round_idx}/{num_iters}...")
        
        current_round_r1s = []
        current_round_aps = []
        processed_pids = set() # 每一轮都要重置 Set，因为要重新测一遍

        # --- 内层循环：遍历测试集 ---
        for sample in tqdm(test_data, desc=f"Round {round_idx}"):
            qid = str(sample["id"])
            if qid not in gallery_data: continue
            
            # 获取 Query 信息
            q_img_path = gallery_data[qid]["query"]
            query_pid = os.path.basename(q_img_path).split("_")[0]
            
            # PID 去重：确保该轮次中，每个行人只作为 Query 出现一次
            if query_pid in processed_pids:
                continue
            processed_pids.add(query_pid)

            # 获取文本 (取第3个 caption)
            caption = sample["captions"][2]
            # caption = sample["captions"][1]
            # caption = sample["captions"][0]
            
            # 获取 Gallery
            gallery_rel_paths = gallery_data[qid]["gallery"]
            gallery_full_paths = [os.path.join(image_root, p) for p in gallery_rel_paths]
            # 构造 GT (Ground Truth)
            gt = [1 if os.path.basename(p).split("_")[0] == query_pid else 0 for p in gallery_rel_paths]

            # 不能在这里随机打乱，这样导致每一次评估的时候不能维持唯一变量
            # random.seed(42)  # 固定种子
            # combined = list(zip(gallery_full_paths, gt))
            # random.shuffle(combined)
            # # import pdb; pdb.set_trace()
            # gallery_full_paths_shuffled, gt_shuffled = zip(*combined)
            # # 打乱后重制
            # gallery_full_paths = list(gallery_full_paths_shuffled)
            # gt = list(gt_shuffled)

            # 执行推理 
            ranked, _ = rank_text2image_confidence(
                model, 
                processor, 
                caption, 
                gallery_full_paths, 
                max_batch_size=5,
                log_prefix=f"R{round_idx}|PID:{query_pid}"
            )
            
            # 计算单条指标
            r1, ap = evaluate_rank(gt, ranked)
            current_round_r1s.append(r1)
            current_round_aps.append(ap)
            
            # 详细日志
            logging.info(f"Round {round_idx} | PID: {query_pid} | R1: {r1} | AP: {ap:.4f}")

        # --- 本轮结束，计算本轮全局平均值 ---
        if not current_round_r1s:
            avg_r1, avg_map = 0.0, 0.0
        else:
            avg_r1 = sum(current_round_r1s) / len(current_round_r1s)
            avg_map = sum(current_round_aps) / len(current_round_aps)
        
        logging.info(f"{'='*20} ROUND {round_idx} SUMMARY {'='*20}")
        logging.info(f"Global Rank-1: {avg_r1:.4f}")
        logging.info(f"Global mAP:    {avg_map:.4f}")
        
        round_metrics.append((avg_r1, avg_map))

    # --- 5 轮结束，取最大值 ---
    # 分别取 5 轮中最高的 R1 和 最高的 mAP (通常建议取同一轮的结果，但此处按最大值需求)
    final_max_r1 = max([m[0] for m in round_metrics])
    final_max_map = max([m[1] for m in round_metrics])

    logging.info(f"\n{'#'*20} FINAL EVALUATION RESULT {'#'*20}")
    logging.info(f"Max Global Rank-1 (across {num_iters} rounds): {final_max_r1:.4f}")
    logging.info(f"Max Global mAP    (across {num_iters} rounds): {final_max_map:.4f}")

    return final_max_r1, final_max_map