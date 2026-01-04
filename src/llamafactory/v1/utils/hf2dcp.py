# convert_hf_to_dcp.py
import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM


def convert(model_path, save_path):
    print(f"Loading HF model from {model_path}...")
    # 这里必须完整加载到 CPU，无法避免
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.bfloat16)

    print(f"Saving to DCP format at {save_path}...")
    # DCP 保存的是 State Dict，它会自动处理目录结构
    dcp.save(model.state_dict(), checkpoint_id=save_path)
    print("Done!")


if __name__ == "__main__":
    convert("./Qwen3-8B", "./Qwen3-8B-DCP")
