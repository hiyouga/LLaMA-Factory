import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from get_test_datasets import load_pstp_test
from baseline_filter_top50 import baseline_filter_50
from eval_text2image import run_iterative_eval

# --- 配置区域 ---
CAPTION_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_caption_all_qwen.json"
IMAGE_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs"
MODEL_PATH = "/home/wangrui/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
CACHE_FILE = "/home/wangrui/code/LLaMA-Factory/output/reid_resnet50_cache/pstp_test_gallery_10_shuffle.json"


def main():
    # 1. 加载数据
    print(">>> Loading Datasets...")
    test_data = load_pstp_test(CAPTION_PATH)

    # 2. 准备 Gallery (如果不存在则生成)
    if not os.path.exists(CACHE_FILE):
        print(">>> Cache not found. Running Baseline Filter (ResNet50)...")
        baseline_filter_50(test_data, IMAGE_PATH, CACHE_FILE)
    else:
        print(f">>> Using existing cache: {CACHE_FILE}")

    # 3. 加载 VLM
    print(f">>> Loading Model from {MODEL_PATH}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 4. 执行 5 轮外层循环评估
    print(">>> Starting 5-Round Iterative Evaluation...")
    final_r1, final_map = run_iterative_eval(
        model, 
        processor, 
        test_data, 
        IMAGE_PATH, 
        CACHE_FILE, 
        num_iters=5
    )

    print("\n" + "="*50)
    print(f"FINAL RESULT (Max of 5 Rounds):")
    print(f"Rank-1: {final_r1:.4f}")
    print(f"mAP   : {final_map:.4f}")
    print("="*50)
    print("Check 'eval_process.log' for detailed logs.")

if __name__ == "__main__":
    main()

