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

import os

import torch


REWARD_HEAD_WEIGHTS_NAME = "reward_head.bin"


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def has_reward_head(model) -> bool:
    model_to_check = _unwrap_model(model)
    return hasattr(model_to_check, "reward_head")


def strip_reward_head_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove reward-head parameters from a model state dict before save_pretrained()."""
    filtered_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if "reward_head" in key.split("."):
            continue
        filtered_state_dict[key] = value
    return filtered_state_dict


def save_reward_head(model, output_dir: str) -> bool:
    model_to_save = _unwrap_model(model)
    reward_head = getattr(model_to_save, "reward_head", None)
    if reward_head is None:
        return False

    os.makedirs(output_dir, exist_ok=True)
    torch.save(reward_head.state_dict(), os.path.join(output_dir, REWARD_HEAD_WEIGHTS_NAME))
    return True


def load_reward_head(model, ckpt_dir: str, device: torch.device | None = None) -> bool:
    model_to_load = _unwrap_model(model)
    reward_head = getattr(model_to_load, "reward_head", None)
    if reward_head is None:
        return False

    path = os.path.join(ckpt_dir, REWARD_HEAD_WEIGHTS_NAME)
    if not os.path.exists(path):
        return False

    map_location = device if device is not None else "cpu"
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    reward_head.load_state_dict(state_dict, strict=True)
    return True
