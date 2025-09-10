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
    p = argparse.ArgumentParser(description="Convert DCP checkpoint formats")
    p.add_argument("--src", required=True, help="Path to DCP checkpoint directory")
    p.add_argument(
        "--torch-out",
        required=False,
        default=None,
        help="Output .pth path for torch.save compatible checkpoint",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = args.src
    if not os.path.isdir(src):
        print(f"Source DCP directory not found: {src}", file=sys.stderr)
        return 2

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
