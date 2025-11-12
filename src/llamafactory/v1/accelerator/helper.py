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

from functools import lru_cache

import torch


def get_current_accelerator(check_available: bool = True):
    """Get current accelerator.

    Note: this api requires torch>=2.7.0, 2.6 or lower will get an AttributeError or RuntimeError
    """
    if not hasattr(torch, "accelerator"):
        raise RuntimeError("torch.accelerator is not available, please upgrade torch to 2.7.0 or higher.")

    accelerator = torch.accelerator.current_accelerator(check_available=check_available)
    if accelerator is None:
        return torch.device("cpu")

    return accelerator


@lru_cache
def is_torch_npu_available():
    return get_current_accelerator().type == "npu"


@lru_cache
def is_torch_cuda_available():
    return get_current_accelerator().type == "cuda"


@lru_cache
def is_torch_xpu_available():
    return get_current_accelerator().type == "xpu"


@lru_cache
def is_torch_mps_available():
    return get_current_accelerator().type == "mps"
