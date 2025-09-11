"""Distributed Checkpoint conversion helpers.

Utilities to convert a Distributed Checkpoint (DCP) directory into a single
torch.save() file using PyTorch's documented format_utils APIs.

Example:
  python -m LLaMA-Factory.scripts.dcp_convert \
      --src output/qwen3_full_sft_fsdp/checkpoint-5 \
      --torch-out exports/qwen3_full_sft_fsdp.pth

After creating a .pth file, you can load it into a plain HF model and
call save_pretrained() if needed.
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert/merge DCP checkpoints")
    p.add_argument("--src", required=True, help="Path to DCP checkpoint directory")
    p.add_argument(
        "--torch-out",
        required=False,
        default=None,
        help="Output .pth path for torch.save compatible checkpoint",
    )
    p.add_argument(
        "--hf-out",
        required=False,
        default=None,
        help="Output directory for Hugging Face save_pretrained export",
    )
    p.add_argument(
        "--base",
        required=False,
        default=None,
        help="Base model ID or local path to instantiate architecture for HF export (e.g., Qwen/Qwen3-8B)",
    )
    p.add_argument(
        "--dtype",
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for instantiating the HF model when merging (default: bfloat16)",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoConfig/AutoModel for custom architectures",
    )
    p.add_argument(
        "--fill-missing",
        action="store_true",
        help=(
            "Do not error on missing checkpoint keys; keep base model weights for missing entries and report counts. "
            "Also forces a resilient merge path that bypasses direct DCP load."
        ),
    )
    return p.parse_args()


def _to_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _merge_dcp_to_hf(
    src: str,
    out_dir: str,
    base: str,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    fill_missing: bool = False,
) -> int:
    import os

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Import the AppState from the training code to leverage forest-aware restore
    try:
        from llamafactory.train.checkpoint_manager import AppState  # type: ignore
    except Exception as e:
        print(f"Failed to import AppState from checkpoint_manager: {e}", file=sys.stderr)
        return 4

    try:
        import torch.distributed.checkpoint as dcp
    except Exception as e:
        print(f"PyTorch DCP unavailable: {e}", file=sys.stderr)
        return 5

    cfg = AutoConfig.from_pretrained(base, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code, torch_dtype=_to_dtype(dtype))

    def _fallback_merge() -> int:
        # Fallback: convert to torch.save and merge by FQN into a plain HF model
        try:
            import tempfile

            import torch
            from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        except Exception as e2:
            print(f"Fallback unavailable: {e2}", file=sys.stderr)
            return 7

        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, "checkpoint.pth")
            dcp_to_torch_save(src, tmp_path)
            blob = torch.load(tmp_path, map_location="cpu")
        # Descend into structure: expect top-level 'app' then 'model'
        root = blob
        if isinstance(root, dict) and "app" in root and isinstance(root["app"], dict):
            root = root["app"]
        model_blob = root.get("model") if isinstance(root, dict) else None
        if not isinstance(model_blob, dict):
            print("Unsupported checkpoint structure; expected a dict under app.model", file=sys.stderr)
            return 8

        # Flatten forest if present, with prefixing of submodule name
        flat: dict[str, torch.Tensor] = {}
        if model_blob.get("__fsdp_forest__"):
            for prefix, sub in model_blob.items():
                if prefix.startswith("__"):
                    continue
                if isinstance(sub, dict):
                    for kk, vv in sub.items():
                        if isinstance(vv, torch.Tensor):
                            key = kk if (not prefix or kk.startswith(prefix + ".")) else f"{prefix}.{kk}"
                            flat[key] = vv
                else:
                    print(f"Warning: unexpected non-dict entry under forest key '{prefix}'", file=sys.stderr)
        else:
            for kk, vv in model_blob.items():
                if isinstance(vv, torch.Tensor):
                    flat[kk] = vv

        msd = model.state_dict()
        assign_count = 0
        for k, v in flat.items():
            if k in msd:
                try:
                    msd[k].copy_(v.to(msd[k].dtype))
                    assign_count += 1
                except Exception:
                    pass
        missing, unexpected = model.load_state_dict(msd, strict=False)
        print(
            f"Merged tensors: {assign_count}, Filled from base (missing): {len(missing)}, Unused tensors: {len(unexpected)}",
            file=sys.stderr,
        )
        return 0

    if fill_missing:
        code = _fallback_merge()
        if code != 0:
            return code
    else:
        app = AppState(model, optimizer=None, scheduler=None)
        state = {"app": app}
        # Non-distributed load: no process group provided
        try:
            dcp.load(state_dict=state, checkpoint_id=src)
        except Exception as e:
            print(f"DCP load failed ({e}). Falling back to torch.save merge...", file=sys.stderr)
            code = _fallback_merge()
            if code != 0:
                return code

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code)
        tok.save_pretrained(out_dir)
    except Exception as e:
        print(f"Warning: failed to save tokenizer: {e}", file=sys.stderr)
    cfg.save_pretrained(out_dir)
    print(f"Wrote Hugging Face model to: {out_dir}")
    return 0


def main() -> int:
    args = parse_args()
    src = args.src
    if not os.path.isdir(src):
        print(f"Source DCP directory not found: {src}", file=sys.stderr)
        return 2

    ret: int | None = None
    if args.torch_out:
        try:
            from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        except Exception as e:
            print(f"PyTorch DCP format_utils unavailable: {e}", file=sys.stderr)
            return 3
        out_path = args.torch_out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        dcp_to_torch_save(src, out_path)
        print(f"Wrote torch.save checkpoint: {out_path}")
        ret = 0

    if args.hf_out:
        if not args.base:
            print("--base is required when using --hf-out", file=sys.stderr)
            return 6
        code = _merge_dcp_to_hf(
            src,
            args.hf_out,
            args.base,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            fill_missing=args.fill_missing,
        )
        ret = code if code != 0 else (ret or 0)

    if not args.torch_out and not args.hf_out:
        print("Nothing to do. Specify --torch-out and/or --hf-out.", file=sys.stderr)
        return 1

    return ret or 0


if __name__ == "__main__":
    raise SystemExit(main())
