"""
功能：手动将 "Stage 1 的 Vision Encoder" 和 "原始的 Qwen2.5-VL Decoder" 拼合成一个新的基础模型。作为第二阶段的模型
的初始化模型。
说明：Stage 1 训练完成后，得到一个视觉编码器已经训练好的模型，并且已经lora合并进去了，只针对retpreid数据集
其他两个数据集也直接使用该模型推理即可 
"""


import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq

# 路径配置（请根据你的实际路径修改）
base_model_path = "/home/wangrui/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"  # 原始模型
stage1_checkpoint = "/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_with_score" # Stage 1 训练后的合并模型
save_path = "/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_stage2_init_model" # Stage 2 的启动模型保存路径

# 我们之前的dpo实验 或者是 微调实验都是针对原始的base_model_path微调的，并没有使用到stage1的decoder，

print("Step 1: 正在加载原始模型 (Base Model for Decoder)...")
# 使用 AutoModel + trust_remote_code 解决 qwen2_5_vl 架构识别问题
model_base = AutoModelForVision2Seq.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True  # <--- 必须开启，否则无法正确加载 Qwen2.5-VL 结构
)

print("Step 2: 正在加载 Stage 1 合并模型 (Stage 1 for Encoder)...")
# 确保这里加载的是已经 Merge 过的全量模型，而不是 LoRA Adapter 文件夹
model_stage1 = AutoModelForVision2Seq.from_pretrained(
    stage1_checkpoint,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True
)

print("Step 3: 正在移植 Visual Encoder 权重...")
# 将 Stage 1 训练好的视觉塔权重，覆盖到 Base 模型的视觉塔上
# Qwen2.5-VL 的视觉模块通常依然命名为 'visual'，但也可能变动，这里做个检查
if hasattr(model_base, "visual"):
    model_base.visual.load_state_dict(model_stage1.visual.state_dict())
else:
    # 防御性编程：如果官方代码修改了变量名（例如 model.model.vision_model）
    print("Warning: 未找到 'visual' 属性，尝试自动查找视觉模块...")
    # 打印所有模块名帮助调试（如果上面报错的话）
    # print(model_base) 
    raise AttributeError("无法找到 visual 模块，请检查 model 结构")

print(f"Step 4: 正在保存 Stage 2 初始化模型到 {save_path} ...")
model_base.save_pretrained(save_path)

print("Step 5: 保存 Processor 配置...")
processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
processor.save_pretrained(save_path)

print("✅ 完成！请在 Stage 2 YAML 中将 model_name_or_path 指向该保存路径。")