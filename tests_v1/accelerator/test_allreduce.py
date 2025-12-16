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


import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llamafactory.v1.accelerator.helper import ReduceOp, all_reduce, is_torch_cuda_available, is_torch_npu_available
from llamafactory.v1.utils.utils import find_available_port


def _dist_worker(rank, world_size):
    if is_torch_cuda_available():
        backend = "nccl"
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
    elif is_torch_npu_available():
        backend = "hccl"
        device = torch.device(f"npu:{rank}")
        torch.npu.set_device(rank)
    else:
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    # --------------------
    # Test all_reduce SUM
    # --------------------
    y = torch.tensor(rank + 1.0, device=device)
    y_sum = all_reduce(y.clone(), op=ReduceOp.SUM)
    assert y_sum.item() == 3.0

    # --------------------
    # Test all_reduce MEAN
    # --------------------
    y_mean = all_reduce(y.clone(), op=ReduceOp.MEAN)
    assert y_mean.item() == pytest.approx(1.5)

    # --------------------
    # Test all_reduce MAX
    # --------------------
    y_max = all_reduce(y.clone(), op=ReduceOp.MAX)
    assert y_max.item() == 2.0

    dist.destroy_process_group()


@pytest.mark.runs_on(["npu", "cuda"])
@pytest.mark.require_distributed(2)
def test_distributed_ops(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(find_available_port()))
    WORLD_SIZE = 2
    mp.spawn(
        _dist_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )


@pytest.mark.runs_on(["npu", "cuda"])
@pytest.mark.require_distributed(4)
def test_required_multi():
    # test require_distributed mark ok
    pass


@pytest.mark.runs_on(["npu", "cuda"])
@pytest.mark.require_distributed(999)
def test_required_invalid():
    # test require_distributed mark not ok,
    raise RuntimeError(
        "this case should not be run, please check whether the require_distributed mark implementation is correct"
    )
