# convert_hf_to_dcp.py
import argparse

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument("--hf_path", type=str, required=True)
parser.add_argument("--dcp_path", type=str, required=True)
args = parser.parse_args()

def convert(model_path, save_path):
    print(f"Loading HF model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.bfloat16)

    print(f"Saving to DCP format at {save_path}...")
    dcp.save(model.state_dict(), checkpoint_id=save_path)
    print("Done!")


if __name__ == "__main__":
    convert(args.hf_path, args.dcp_path)
