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
import socket

import torch

from llamafactory.v1.accelerator.helper import (
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)


def find_available_port() -> int:
    r"""Find an available port on the local machine."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def get_current_device() -> "torch.device":
    r"""Get the current available device."""
    if is_torch_xpu_available():
        device = "xpu:{}".format(os.getenv("LOCAL_RANK", "0"))
    elif is_torch_npu_available():
        device = "npu:{}".format(os.getenv("LOCAL_RANK", "0"))
    elif is_torch_mps_available():
        device = "mps:{}".format(os.getenv("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.getenv("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def get_device_count() -> int:
    r"""Get the number of available devices."""
    if is_torch_xpu_available():
        return torch.xpu.device_count()
    elif is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_mps_available():
        return torch.mps.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    r"""Check if the environment variable is enabled."""
    return os.getenv(env_var, default).lower() in ["true", "y", "1"]


if __name__ == "__main__":
    print(find_available_port())
