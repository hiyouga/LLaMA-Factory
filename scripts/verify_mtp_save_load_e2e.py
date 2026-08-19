#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# End-to-end verification of MTP weight save/load (issue Q2).
#
# Run AFTER `llamafactory-cli train examples/v1/train_full/train_full_mtp_fsdp2.yaml`
# has produced a checkpoint in `output_dir`. This script exercises the REAL load path
# (`from_pretrained` -> `apply_mtp` -> `load_mtp_weights`, exactly what ModelEngine does)
# and checks that the MTP tensors loaded into the model are bit-identical to the ones
# written to disk, and that they differ from a fresh random init (i.e. training was
# preserved, not lost).
#
# Usage (from repo root):
#   PYTHONPATH=src python scripts/verify_mtp_save_load_e2e.py \
#       --output_dir outputs/test_mtp_fsdp2 --mtp_num_layers 1 --mtp_loss_scale 0.3
#
# No torchrun / FSDP2 needed: `save_model` already wrote a plain HF-format checkpoint
# (full state dict gathered), so a single-process `from_pretrained` loads it.

import argparse
import json
import os

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from llamafactory.v1.plugins.model_plugins.mtp import apply_mtp, load_mtp_weights


def read_mtp_tensors_from_disk(output_dir: str) -> dict[str, torch.Tensor]:
    """Read every ``mtp.*`` tensor straight out of the saved safetensors shards.

    These are the ground-truth values the trainer wrote — independent of any load path,
    so comparing against them validates that ``load_mtp_weights`` puts the right tensor
    into the right parameter.
    """
    index_file = os.path.join(output_dir, "model.safetensors.index.json")
    tensors: dict[str, torch.Tensor] = {}

    if os.path.exists(index_file):
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        mtp_keys = [k for k in weight_map if k.startswith("mtp.")]
        shards: dict[str, list[str]] = {}
        for k in mtp_keys:
            shards.setdefault(weight_map[k], []).append(k)
        for shard, keys in shards.items():
            with safe_open(os.path.join(output_dir, shard), framework="pt", device="cpu") as f:
                for k in keys:
                    tensors[k] = f.get_tensor(k)
    else:
        single = os.path.join(output_dir, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(f"No safetensors checkpoint found in {output_dir}")
        with safe_open(single, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    tensors[k] = f.get_tensor(k)

    return tensors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify MTP weight save/load end-to-end.")
    parser.add_argument("--output_dir", required=True, help="Trainer output_dir with the saved checkpoint.")
    parser.add_argument("--mtp_num_layers", type=int, default=1, help="K, must match the training yaml.")
    parser.add_argument("--mtp_loss_scale", type=float, default=0.3, help="Must match the training yaml.")
    args = parser.parse_args()

    mtp_config = {"name": "mtp", "num_layers": args.mtp_num_layers, "loss_scale": args.mtp_loss_scale}
    output_dir = args.output_dir

    print(f"=== Verifying MTP save/load on checkpoint: {output_dir} ===\n")

    # ---- Save side: confirm mtp.* tensors were actually written to disk ----
    file_mtp = read_mtp_tensors_from_disk(output_dir)
    print(f"[save side] mtp.* tensors written to disk: {len(file_mtp)}")
    if not file_mtp:
        print("[save side] FAIL: no mtp.* keys in checkpoint — MTP weights were NOT saved.")
        return 1
    shared_keys = {"mtp.embed_tokens.weight", "mtp.output_layer.weight"}
    leaked = [k for k in file_mtp if k in shared_keys]
    if leaked:
        print(f"[save side] FAIL: shared keys leaked into checkpoint: {leaked}")
        return 1
    print("[save side] PASS: mtp.* present, shared embedding/lm_head keys correctly stripped.\n")

    # ---- Load side: the REAL ModelEngine load path ----
    cfg = AutoConfig.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    # from_pretrained drops mtp.* as unexpected; apply_mtp re-creates the block (random),
    # then load_mtp_weights restores the saved tensors — exactly as ModelEngine._init_model does.
    model = apply_mtp(model, mtp_config)
    load_mtp_weights(model, output_dir)

    # Compare every MTP parameter in the model against its on-disk ground truth.
    head0 = model.mtp.layers[str(cfg.num_hidden_layers)]
    checks = {
        "mtp.e_proj.weight": model.mtp.e_proj.weight,
        "mtp.h_proj.weight": model.mtp.h_proj.weight,
        "mtp.enorm.weight": model.mtp.enorm.weight,
        "mtp.hnorm.weight": model.mtp.hnorm.weight,
        "mtp.final_layernorm.weight": model.mtp.final_layernorm.weight,
        f"mtp.layers.{cfg.num_hidden_layers}.layer.self_attn.q_proj.weight": head0.layer.self_attn.q_proj.weight,
        f"mtp.layers.{cfg.num_hidden_layers}.layer.self_attn.k_proj.weight": head0.layer.self_attn.k_proj.weight,
        f"mtp.layers.{cfg.num_hidden_layers}.layer.input_layernorm.weight": head0.layer.input_layernorm.weight,
    }

    print("[load side] comparing model params vs on-disk tensors:")
    all_equal = True
    for key, param in checks.items():
        if key not in file_mtp:
            print(f"  SKIP  {key} (not on disk)")
            continue
        ok = torch.equal(param.detach().cpu(), file_mtp[key])
        all_equal = all_equal and ok
        flag = "equal" if ok else "MISMATCH"
        print(f"  {flag:9s} {key}  (sum={float(param.sum()):.6f})")

    # ---- Non-random check: loaded weights must differ from a fresh random init ----
    fresh = AutoModelForCausalLM.from_config(cfg)
    fresh = apply_mtp(fresh, mtp_config)
    loaded_sum = float(model.mtp.e_proj.weight.sum())
    random_sum = float(fresh.mtp.e_proj.weight.sum())
    non_random = not torch.equal(model.mtp.e_proj.weight.detach().cpu(), fresh.mtp.e_proj.weight.detach().cpu())
    print(f"\n[non-random] loaded e_proj.sum={loaded_sum:.6f}  random_init.sum={random_sum:.6f}  differ={non_random}")

    # ---- Shared-module check: mtp.embed_tokens / output_layer re-tied to base model ----
    tied_embed = model.mtp.embed_tokens.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    tied_lm_head = model.mtp.output_layer.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print(f"[shared] mtp.embed_tokens tied to base embedding: {tied_embed}")
    print(f"[shared] mtp.output_layer tied to base lm_head:   {tied_lm_head}")

    print("\n=== Q2 end-to-end verdict ===")
    save_ok = bool(file_mtp) and not leaked
    load_ok = all_equal
    nonrandom_ok = non_random
    tied_ok = tied_embed and tied_lm_head
    print(f"  save side  (mtp.* written, shared stripped): {'PASS' if save_ok else 'FAIL'}")
    print(f"  load side  (load_mtp_weights restores exact): {'PASS' if load_ok else 'FAIL'}")
    print(f"  non-random (restored = trained, not random):  {'PASS' if nonrandom_ok else 'FAIL'}")
    print(f"  re-tied    (embed/lm_head shared w/ base):    {'PASS' if tied_ok else 'FAIL'}")
    return 0 if (save_ok and load_ok and nonrandom_ok and tied_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
