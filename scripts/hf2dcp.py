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

"""Convert a HuggingFace model to DCP checkpoint format.

Usage:
  python scripts/hf2dcp.py convert --hf_path=/path/to/hf --dcp_path=/path/to/dcp

Arguments:
  hf_path: Path to the HuggingFace model directory.
  dcp_path: Output path (directory) for DCP checkpoint.
"""

import fire
import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM


def convert(hf_path: str, dcp_path: str) -> None:
    """Convert HF model weights to DCP.

    Args:
        hf_path: HuggingFace model directory.
        dcp_path: Output path (directory) for DCP checkpoint.
    """
    if not hf_path or not dcp_path:
        raise ValueError("Both 'hf_path' and 'dcp_path' are required.")

    print(f"Loading HF model from {hf_path}...")
    model = AutoModelForCausalLM.from_pretrained(hf_path, device_map="cpu", torch_dtype=torch.bfloat16)

    print(f"Saving to DCP format at {dcp_path}...")
    dcp.save(model.state_dict(), checkpoint_id=dcp_path)
    print("Done!")


def help() -> None:
    """Show help message."""
    print(__doc__)


if __name__ == "__main__":
    fire.Fire({"convert": convert, "help": help, "--convert": convert})
