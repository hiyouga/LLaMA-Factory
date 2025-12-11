#!/usr/bin/env python3
"""
Convert FSDP sharded model to HuggingFace multi-shard (split) format.
"""
import os
import torch
import torch.distributed.tensor
from pathlib import Path
import shutil
import json

def inspect_shard(shard_path):
    """Inspect a single shard to understand its structure"""
    print(f"\nInspecting shard: {shard_path}")
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    
    if isinstance(shard, dict):
        print(f"  Type: dict with {len(shard)} keys")
        for key in list(shard.keys())[:5]:
            value = shard[key]
            if isinstance(value, torch.Tensor):
                print(f"    {key}: Tensor {value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"    {key}: dict with {len(value)} keys")
            else:
                print(f"    {key}: {type(value)}")
    else:
        print(f"  Type: {type(shard)}")
    
    return shard

def extract_flat_state_dict(shard):
    """Extract a flat state dict from FSDP shard, handling DTensor"""
    state_dict = {}
    
    def process_value(value):
        """Convert DTensor to regular tensor if needed"""
        if hasattr(value, '_local_tensor'):
            # DTensor - extract local tensor
            return value._local_tensor
        elif hasattr(value, 'to_local'):
            # DTensor with to_local method
            return value.to_local()
        elif isinstance(value, torch.Tensor):
            return value
        else:
            return value
    
    if isinstance(shard, dict):
        # Check for common FSDP structures
        if 'state' in shard and isinstance(shard['state'], dict):
            # FSDP wrapped with state key
            for key, value in shard['state'].items():
                state_dict[key] = process_value(value)
        elif 'model_state_dict' in shard:
            for key, value in shard['model_state_dict'].items():
                state_dict[key] = process_value(value)
        else:
            # Direct state dict
            for key, value in shard.items():
                if isinstance(value, (torch.Tensor, type(None))) or hasattr(value, '_local_tensor'):
                    state_dict[key] = process_value(value)
    
    return state_dict

def split_state_dict_to_shards(state_dict, num_output_shards=8, prefer_split_on_dim1=False):
    """
    Split a consolidated (unified) state_dict into many shards for HuggingFace split/model shard checkpointing.
    - num_output_shards: number of output files to write.
    Returns: list of dicts (shards)
    """
    import math
    key_list = list(state_dict.keys())
    num_keys = len(key_list)
    # Make nearly equal number of keys per shard, but try to be parameter-count aware
    # We'll just group parameters in order, to start (as HuggingFace does for massive models)
    shard_dicts = [{} for _ in range(num_output_shards)]
    size_per_shard = [0 for _ in range(num_output_shards)]
    # To get better balance: assign each parameter tensor to the smallest shard so far
    for key in key_list:
        param = state_dict[key]
        sz = param.numel() if isinstance(param, torch.Tensor) else 1
        target_shard = size_per_shard.index(min(size_per_shard))
        shard_dicts[target_shard][key] = param
        size_per_shard[target_shard] += sz
    return shard_dicts

def consolidate_fsdp_shards_and_split(fsdp_dir, output_dir, num_shards=4, num_output_shards=8):
    """
    Consolidate FSDP model shards into multiple HuggingFace checkpoint shards (split).
    """
    fsdp_dir = Path(fsdp_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading FSDP shards from: {fsdp_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Will output {num_output_shards} shards (files)")

    # Load config from huggingface subfolder
    hf_config_dir = fsdp_dir / "huggingface"
    if not hf_config_dir.exists():
        raise ValueError(f"HuggingFace config directory not found: {hf_config_dir}")
    
    print(f"\nCopying tokenizer and config files...")
    for file in hf_config_dir.glob("*"):
        if file.is_file():
            shutil.copy2(file, output_dir / file.name)
            print(f"  Copied: {file.name}")
    
    # Inspect first shard
    shard_path = fsdp_dir / f"model_world_size_{num_shards}_rank_0.pt"
    sample_shard = inspect_shard(shard_path)
    
    # Load all FSDP shards
    print(f"\nLoading {num_shards} FSDP model shards...")
    state_dicts = []
    for rank in range(num_shards):
        shard_path = fsdp_dir / f"model_world_size_{num_shards}_rank_{rank}.pt"
        if not shard_path.exists():
            raise ValueError(f"Shard not found: {shard_path}")
        print(f"  Loading shard {rank}...")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        flat_dict = extract_flat_state_dict(shard)
        state_dicts.append(flat_dict)
        print(f"    Extracted {len(flat_dict)} parameters")
    
    # Consolidate state dicts
    print("\nConsolidating model state dict...")
    consolidated_state_dict = {}
    
    # Collect all unique keys
    all_keys = set()
    for shard_dict in state_dicts:
        all_keys.update(shard_dict.keys())
    print(f"  Found {len(all_keys)} unique parameter keys")
    for key in sorted(all_keys):
        # Collect all values for this key across shards
        values = []
        for shard_dict in state_dicts:
            if key in shard_dict:
                val = shard_dict[key]
                if val is not None:
                    values.append(val)
        if len(values) == 0:
            continue
        elif len(values) == 1:
            # Parameter only in one shard
            consolidated_state_dict[key] = values[0]
        else:
            # Parameter sharded across multiple ranks
            # Check if all values are identical (replicated parameter)
            if all(torch.equal(values[0], v) for v in values[1:]):
                # Replicated - just use one copy
                consolidated_state_dict[key] = values[0]
            else:
                # Sharded - try to concatenate
                # FSDP typically shards along dim 0
                try:
                    consolidated_state_dict[key] = torch.cat(values, dim=0)
                except Exception as e:
                    print(f"  Warning: Could not concatenate {key}, using first shard. Error: {e}")
                    consolidated_state_dict[key] = values[0]
    print(f"\nConsolidated state dict has {len(consolidated_state_dict)} keys")
    
    # Convert float32 to bfloat16 for every param, as in HuggingFace's vllm expectations
    for key in list(consolidated_state_dict.keys()):
        v = consolidated_state_dict[key]
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            consolidated_state_dict[key] = v.to(torch.bfloat16)

    # Split state dict into shard dicts
    split_dicts = split_state_dict_to_shards(consolidated_state_dict, num_output_shards=num_output_shards)

    # Save each shard as safetensors and create "weight_map" for HuggingFace
    try:
        from safetensors.torch import save_file
        weight_map = {}
        filenames = []
        for i, shard_dict in enumerate(split_dicts):
            filename = f"model-{i+1:05d}-of-{num_output_shards:05d}.safetensors"
            path = output_dir / filename
            save_file(shard_dict, path)
            print(f"  Saved {filename} with {len(shard_dict)} tensors")
            # Add every param in this shard's dict to weight_map
            for key in shard_dict:
                weight_map[key] = filename
            filenames.append(filename)
        
        # Create model index file
        index_file = {
            "metadata": {"total_size": sum(p.numel() * p.element_size() for p in consolidated_state_dict.values())},
            "weight_map": weight_map
        }
        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index_file, f, indent=2)
        print("  Saved model.safetensors.index.json")
    except ImportError:
        # Fall back to PyTorch sharded .bin files
        weight_map = {}
        filenames = []
        for i, shard_dict in enumerate(split_dicts):
            filename = f"pytorch_model-{i+1:05d}-of-{num_output_shards:05d}.bin"
            path = output_dir / filename
            torch.save(shard_dict, path)
            print(f"  Saved {filename} with {len(shard_dict)} tensors")
            for key in shard_dict:
                weight_map[key] = filename
            filenames.append(filename)
        # Create model index file
        index_file = {
            "metadata": {"total_size": sum(p.numel() * p.element_size() for p in consolidated_state_dict.values())},
            "weight_map": weight_map
        }
        with open(output_dir / "pytorch_model.bin.index.json", "w") as f:
            json.dump(index_file, f, indent=2)
        print("  Saved pytorch_model.bin.index.json")
    
    print("\n✓ Conversion complete!")
    print(f"Model saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    import sys

    # Configuration
    FSDP_DIR = "/home/ubuntu/research_nfs/jasonqi_weights/llm_as_a_judge_t_bench/20251124_222213/global_step_100/policy"
    OUTPUT_DIR = "/home/ubuntu/research_nfs/jasonqi_weights/llm_as_a_judge_t_bench/20251124_222213_hf"
    NUM_SHARDS = 4           # Number of input FSDP shards (original)
    NUM_OUTPUT_SHARDS = 8    # Number of HuggingFace output shards/files

    print("=" * 80)
    print("FSDP to HuggingFace Converter (Multi-shard)")
    print("=" * 80)

    try:
        consolidate_fsdp_shards_and_split(FSDP_DIR, OUTPUT_DIR, NUM_SHARDS, NUM_OUTPUT_SHARDS)
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

