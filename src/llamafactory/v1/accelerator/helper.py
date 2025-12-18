# Copyright 2025 Bytedance Ltd. and the LlamaFactory team.
#
# This code is inspired by the Bytedance's VeOmni library.
# https://github.com/ByteDance-Seed/VeOmni/blob/v0.1.4/veomni/utils/dist_utils.py
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
from contextlib import contextmanager
from enum import Enum, unique
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from ..utils.types import ProcessGroup, Tensor, TensorLike


@unique
class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    META = "meta"
    MPS = "mps"
    NPU = "npu"
    XPU = "xpu"


@unique
class ReduceOp(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


def is_distributed() -> bool:
    """Check if distributed environment is available."""
    return os.getenv("RANK") is not None


def get_rank() -> int:
    """Get rank."""
    return int(os.getenv("RANK", "0"))


def get_world_size() -> int:
    """Get world size."""
    return int(os.getenv("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    """Get local rank."""
    return int(os.getenv("LOCAL_RANK", "0"))


def get_local_world_size() -> int:
    """Get local world size."""
    return int(os.getenv("LOCAL_WORLD_SIZE", "1"))


@lru_cache
def get_current_accelerator(check_available: bool = True) -> torch.device:
    """Get current accelerator.

    Note: this api requires torch>=2.7.0, otherwise it will raise an AttributeError or RuntimeError
    """
    if not hasattr(torch, "accelerator"):
        raise RuntimeError("torch.accelerator is not available, please upgrade torch to 2.7.0 or higher.")

    accelerator = torch.accelerator.current_accelerator(check_available=check_available)
    if accelerator is None:
        return torch.device(DeviceType.CPU.value)

    return accelerator


def is_torch_cuda_available():
    return get_current_accelerator().type == DeviceType.CUDA


def is_torch_mps_available():
    return get_current_accelerator().type == DeviceType.MPS


def is_torch_npu_available():
    return get_current_accelerator().type == DeviceType.NPU


def is_torch_xpu_available():
    return get_current_accelerator().type == DeviceType.XPU


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


def all_gather(tensor: Tensor, group: Optional[ProcessGroup] = None) -> Tensor:
    """Gathers the tensor from all ranks and concats them along the first dim."""
    world_size = get_world_size()
    device = get_current_accelerator()
    output_tensor = torch.empty(world_size * tensor.numel(), dtype=tensor.dtype, device=device)
    dist.all_gather_into_tensor(output_tensor, tensor, group=group)
    return output_tensor.view(-1, *tensor.size()[1:])


def all_reduce(data: TensorLike, op: ReduceOp = ReduceOp.MEAN, group: Optional[ProcessGroup] = None) -> TensorLike:
    """Performs all reduce in the given process group."""
    device = get_current_accelerator()
    is_ndarray = isinstance(data, np.ndarray)
    is_tensor = isinstance(data, torch.Tensor)

    if is_ndarray:
        data = torch.from_numpy(data).to(device=device, dtype=torch.float)
    elif not is_tensor:
        data = torch.tensor(data, dtype=torch.float, device=device)

    reduce_ops = {
        ReduceOp.MEAN: dist.ReduceOp.SUM,
        ReduceOp.SUM: dist.ReduceOp.SUM,
        ReduceOp.MAX: dist.ReduceOp.MAX,
        ReduceOp.MIN: dist.ReduceOp.MIN,
    }
    dist.all_reduce(data, op=reduce_ops[op], group=group)
    if op == ReduceOp.MEAN:  # ReduceOp.AVG is not supported by the NPU backend
        data /= dist.get_world_size(group=group)

    if is_tensor:
        return data
    elif is_ndarray:
        return data.cpu().numpy()
    elif data.numel() == 1:
        return data.item()
    else:
        return data.tolist()


@contextmanager
def main_process_first(local_only: bool = True) -> None:
    """A context manager for torch distributed environment to do something on the main process firstly."""
    if get_world_size() > 1:
        is_main_process = get_local_rank() == 0 if local_only else get_rank() == 0
        try:
            if not is_main_process:
                dist.barrier()
            yield
        finally:
            if is_main_process:
                dist.barrier()
    else:
        yield
