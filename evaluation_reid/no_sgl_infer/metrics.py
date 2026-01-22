import numpy as np

def compute_ap(gt, ranked):
    """
    计算单个查询的平均精度 (AP) - 基于累积精确率 (Cumulative Precision)
    gt: 二进制列表，1表示匹配，0表示不匹配
    ranked: 模型排序后的索引列表
    """
    num_pos = sum(gt)
    if num_pos == 0:
        return 0.0

    hit = 0
    sum_precision = 0.0
    
    # 遍历排序列表 (start=1 表示 rank 从 1 开始)
    for r, idx in enumerate(ranked, start=1):
        if gt[idx] == 1:
            hit += 1
            # 累积精确率 = 当前累积正确数 / 当前排名位置
            precision_at_k = hit / r
            sum_precision += precision_at_k
            
    # AP = 累加值 / 该行人的总正样本数
    return sum_precision / num_pos

def evaluate_rank(gt, ranked):
    """
    返回 Rank-1 (0或1) 和 AP 值
    """
    # 检查第一名是否命中
    rank1 = 1 if gt[ranked[0]] == 1 else 0
    ap = compute_ap(gt, ranked)
    return rank1, ap