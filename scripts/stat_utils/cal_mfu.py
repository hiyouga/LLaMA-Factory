# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import fire
import torch
import torch.distributed as dist
from transformers import AutoConfig

from llamafactory.train.tuner import run_exp


try:
    # Prefer OmegaConf for robust YAML/JSON config merging
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OmegaConf = None  # type: ignore


BASE = 2  # gemm (add + mul)


def compute_model_flops(
    model_name_or_path: str,
    total_batch_size: int,
    seq_length: int,
    include_backward: bool = True,
    include_recompute: bool = False,
    include_flashattn: bool = False,
) -> int:
    r"""Calculate the FLOPs of model per forward/backward pass."""
    config = AutoConfig.from_pretrained(model_name_or_path)
    hidden_size = getattr(config, "hidden_size", None)
    vocab_size = getattr(config, "vocab_size", None)
    intermediate_size = getattr(config, "intermediate_size", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

    # mlp module
    mlp_flops_per_token = 3 * BASE * hidden_size * intermediate_size  # up, gate, down
    mlp_flops = total_batch_size * seq_length * num_hidden_layers * mlp_flops_per_token

    # attn projector module
    q_flops_per_token = BASE * hidden_size * hidden_size
    o_flops_per_token = BASE * hidden_size * hidden_size
    k_flops_per_token = BASE * hidden_size * hidden_size * num_key_value_heads // num_attention_heads
    v_flops_per_token = BASE * hidden_size * hidden_size * num_key_value_heads // num_attention_heads
    attn_proj_flops_per_token = q_flops_per_token + o_flops_per_token + k_flops_per_token + v_flops_per_token
    attn_proj_flops = total_batch_size * seq_length * num_hidden_layers * attn_proj_flops_per_token

    # attn sdpa module
    sdpa_flops_per_layer = 2 * BASE * hidden_size * seq_length * seq_length  # (q * k^T) * v
    sdpa_flops = total_batch_size * num_hidden_layers * sdpa_flops_per_layer

    # embedding module
    embedding_flops_per_token = hidden_size * vocab_size
    embedding_flops = total_batch_size * seq_length * embedding_flops_per_token
    if tie_word_embeddings is False:
        embedding_flops *= 2

    non_embedding_flops = mlp_flops + attn_proj_flops + sdpa_flops
    non_embedding_coeff, embedding_coeff = 1, 1
    if include_backward:
        non_embedding_coeff += 2
        embedding_coeff += 2

    if include_recompute:
        non_embedding_coeff += 1

    total_flops = non_embedding_coeff * non_embedding_flops + embedding_coeff * embedding_flops

    if include_flashattn:
        total_flops += sdpa_flops

    return total_flops


def compute_device_flops(world_size: int) -> float:
    r"""Calculate the FLOPs of the device capability per second."""
    device_name = torch.cuda.get_device_name()
    if "H100" in device_name or "H800" in device_name:
        return 989 * 1e12 * world_size
    elif "A100" in device_name or "A800" in device_name:
        return 312 * 1e12 * world_size
    elif "V100" in device_name:
        return 125 * 1e12 * world_size
    elif "4090" in device_name:
        return 98 * 1e12 * world_size
    else:
        raise NotImplementedError(f"Device not supported: {device_name}.")


def calculate_mfu(
    model_name_or_path: str = None,
    batch_size: int = None,
    seq_length: int = None,
    num_steps: int = None,
    finetuning_type: str = None,
    attn: str = None,
    deepspeed_stage: int = None,
    disable_gc: bool = None,
    liger_kernel: bool = None,
    unsloth_gc: bool = None,
    config_path: str = None,
    output_dir: str = os.path.join("saves", "test_mfu"),
) -> float:
    r"""Calculate MFU for given model and hyper-params.

    Usage: python cal_mfu.py --model_name_or_path path_to_model --batch_size 1 --seq_length 1024
    """
    # Build training args, optionally starting from a config file
    if config_path:
        if OmegaConf is None:
            raise RuntimeError(
                "OmegaConf is required for --config_path. Please install omegaconf or omit config_path."
            )
        cfg = OmegaConf.load(config_path)
        # Convert to standard Python container (resolve interpolations)
        args = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        if not isinstance(args, dict):
            raise ValueError("Loaded config must resolve to a dict.")

        # Apply non-intrusive defaults suitable for a quick MFU probe if absent
        args.setdefault("do_train", True)
        args.setdefault("stage", "pt")  # prefer a light pretraining step for MFU probe
        args.setdefault("output_dir", output_dir)
        args.setdefault("logging_strategy", "no")
        args.setdefault("save_strategy", "no")
        args.setdefault("save_only_model", True)
        args.setdefault("overwrite_output_dir", True)
        args.setdefault("bf16", True)

        # CLI overrides (only set if provided)
        if model_name_or_path is not None:
            args["model_name_or_path"] = model_name_or_path
        if attn is not None:
            args["attn"] = attn
        if disable_gc is not None:
            args["disable_gradient_checkpointing"] = disable_gc
        if liger_kernel is not None:
            args["enable_liger_kernel"] = liger_kernel
        if unsloth_gc is not None:
            args["use_unsloth_gc"] = unsloth_gc
        if finetuning_type is not None:
            args["finetuning_type"] = finetuning_type
        if batch_size is not None:
            args["per_device_train_batch_size"] = batch_size
        if seq_length is not None:
            args["cutoff_len"] = seq_length
        if num_steps is not None:
            args["max_steps"] = num_steps
        if output_dir:
            args["output_dir"] = output_dir

        # Optional DeepSpeed stage override if provided and not already a full config object
        if deepspeed_stage in [2, 3] and not isinstance(args.get("deepspeed"), (dict, list)):
            args["deepspeed"] = f"examples/deepspeed/ds_z{deepspeed_stage}_config.json"

        # Validate required fields
        if "model_name_or_path" not in args or not args["model_name_or_path"]:
            raise ValueError("model_name_or_path must be set via config_path or CLI.")
    else:
        # Original lightweight defaults when no config is provided
        # Fills any None values with reasonable defaults
        args = {
            "model_name_or_path": model_name_or_path,
            "attn": attn if attn is not None else "eager",
            "disable_gradient_checkpointing": bool(disable_gc) if disable_gc is not None else False,
            "enable_liger_kernel": bool(liger_kernel) if liger_kernel is not None else False,
            "use_unsloth_gc": bool(unsloth_gc) if unsloth_gc is not None else False,
            "stage": "pt",
            "do_train": True,
            "finetuning_type": finetuning_type if finetuning_type is not None else "lora",
            "dataset": "c4_demo",
            "cutoff_len": seq_length if seq_length is not None else 1024,
            "output_dir": output_dir,
            "logging_strategy": "no",
            "save_strategy": "no",
            "save_only_model": True,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": batch_size if batch_size is not None else 1,
            "max_steps": num_steps if num_steps is not None else 100,
            "bf16": True,
        }
        if deepspeed_stage in [2, 3]:
            args["deepspeed"] = f"examples/deepspeed/ds_z{deepspeed_stage}_config.json"

    run_exp(args)
    if dist.is_initialized():
        dist.barrier()
        world_size = dist.get_world_size()
    else:
        world_size = 1

    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        with open(os.path.join(output_dir, "all_results.json"), encoding="utf-8") as f:
            result = json.load(f)

        # Resolve effective batch size and seq length from merged args if not provided
        eff_batch = batch_size if batch_size is not None else int(args.get("per_device_train_batch_size", 1))
        eff_seq = seq_length if seq_length is not None else int(args.get("cutoff_len", 1024))
        eff_model = model_name_or_path or args.get("model_name_or_path")
        if not eff_model:
            raise ValueError("model_name_or_path must be specified.")

        total_batch_size = eff_batch * world_size
        mfu_value = (
            result["train_steps_per_second"]
            * compute_model_flops(eff_model, total_batch_size, eff_seq)
            / compute_device_flops(world_size)
        )
        print(f"MFU: {mfu_value * 100:.2f}%")


if __name__ == "__main__":
    fire.Fire(calculate_mfu)
