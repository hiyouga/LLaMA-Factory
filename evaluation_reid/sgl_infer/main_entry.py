import os
from get_test_datasets import load_pstp_test
from baseline_filter_top50 import baseline_filter_50
from eval_text2image_v2 import run_iterative_eval

# 配置路径
CAPTION_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_caption_all_qwen.json"
IMAGE_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs"
CACHE_FILE = "/home/wangrui/code/LLaMA-Factory/output/reid_resnet50_cache/pstp_gallery_top50_cache.json"

def main():
    # 1. 加载数据
    test_data = load_pstp_test(CAPTION_PATH)

    # 2. 粗筛 (ResNet-50 填充到 50 张)
    if not os.path.exists(CACHE_FILE):
        print("Starting Baseline Filter (50 images)...")
        baseline_filter_50(test_data, IMAGE_PATH, CACHE_FILE)

    # 3. VLM 评估 (确保此时 SGLang 服务已在后台运行)
    print("Starting VLM Iterative Evaluation...")
    r1, mAP = run_iterative_eval(test_data, IMAGE_PATH, CACHE_FILE, num_iters=5)

if __name__ == "__main__":
    main()