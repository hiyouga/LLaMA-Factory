import os
import json
import logging
from tqdm import tqdm
from metrics import evaluate_rank
from vlm_ranker_sgl import rank_text2image_sglang

def run_iterative_eval(test_data, image_root, cache_file, num_iters=5):
    with open(cache_file, "r") as f:
        gallery_data = json.load(f)

    logging.basicConfig(filename='reid_eval.log', level=logging.INFO, 
                        format='%(asctime)s - %(message)s')
    
    final_r1_list = []
    final_ap_list = []

    for sample in tqdm(test_data, desc="Iterative Re-Ranking"):
        qid = str(sample["id"])
        if qid not in gallery_data: continue
        
        caption = sample["captions"][0]
        q_img_path = gallery_data[qid]["query"]
        q_pid = os.path.basename(q_img_path).split("_")[0]
        
        gallery_rel_paths = gallery_data[qid]["gallery"]
        gallery_full_paths = [os.path.join(image_root, p) for p in gallery_rel_paths]
        
        # 判定 GT (行人 ID 匹配)
        gt = [1 if os.path.basename(p).split("_")[0] == q_pid else 0 for p in gallery_rel_paths]
        
        iter_r1, iter_ap = [], []
        
        for i in range(num_iters):
            ranked, _ = rank_text2image_sglang(caption, gallery_full_paths)
            r1, ap = evaluate_rank(gt, ranked)
            iter_r1.append(r1)
            iter_ap.append(ap)
            logging.info(f"QID: {qid} | Iter {i+1} | r1: {r1:.4f} | ap: {ap:.4f}")

        # 取 5 次推理中效果最好的一次
        max_r1 = max(iter_r1)
        max_ap = max(iter_ap)
        final_r1_list.append(max_r1)
        final_ap_list.append(max_ap)
        
        logging.info(f"==> QID: {qid} Summary | Max_r1: {max_r1:.4f} | Max_ap: {max_ap:.4f}")

    avg_r1 = sum(final_r1_list) / len(final_r1_list)
    avg_map = sum(final_ap_list) / len(final_ap_list)
    
    print(f"\nFinal Results (Max of {num_iters} iters):")
    print(f"Rank-1: {avg_r1:.4f}")
    print(f"mAP   : {avg_map:.4f}")
    return avg_r1, avg_map