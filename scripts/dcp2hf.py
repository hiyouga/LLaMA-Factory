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

"""Convert a DCP checkpoint to HuggingFace model format.

Usage:
  python scripts/dcp2hf.py convert --dcp_path=/path/to/dcp --hf_path=/path/to/hf --config_path=/path/to/config

Arguments:
  dcp_path: Path to the DCP checkpoint directory.
  hf_path: Output path (directory) for HuggingFace model.
  config_path: Path to the HuggingFace model directory containing config.json.
"""

import fire
import torch
import torch.distributed.checkpoint as dcp
import transformers
from transformers import AutoConfig


def convert(dcp_path: str, hf_path: str, config_path: str) -> None:
    """Convert DCP model weights to HF.

    Note: this script is used to convert a DCP checkpoint to HuggingFace model format,
    it will just convert the DCP checkpoint to a HuggingFace model format, for the tokenizer,
    you may need to copy from the original model.

    Args:
        dcp_path: DCP checkpoint directory.
        hf_path: Output path (directory) for HuggingFace model.
        config_path: Path to the HuggingFace model directory containing config.json.
    """
    if not dcp_path or not hf_path or not config_path:
        raise ValueError("All 'dcp_path', 'hf_path', and 'config_path' are required.")

    print(f"Loading config from {config_path}...")
    config = AutoConfig.from_pretrained(config_path)
    architectures = getattr(config, "architectures", [])
    if architectures:
        model_cls = getattr(transformers, architectures[0], transformers.AutoModelForCausalLM)
    else:
        model_cls = transformers.AutoModelForCausalLM

    print("Initializing model on CPU...")
    model = model_cls(config).to(torch.bfloat16)

    print(f"Loading DCP from {dcp_path}...")
    state_dict = model.state_dict()
    dcp.load(state_dict, checkpoint_id=dcp_path)
    model.load_state_dict(state_dict)

    print(f"Saving to HF format at {hf_path}...")
    model.save_pretrained(hf_path)
    config.save_pretrained(hf_path)
    print("Done!")


def help() -> None:
    """Show help message."""
    print(__doc__)


if __name__ == "__main__":
    fire.Fire({"convert": convert, "help": help, "--convert": convert})
