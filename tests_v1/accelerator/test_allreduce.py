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

from llamafactory.v1.accelerator.helper import ReduceOp, all_reduce, is_torch_npu_available


def _dist_worker(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
    )

    # --------------------
    # Test all_reduce SUM
    # --------------------
    y = torch.tensor(rank + 1.0, device="npu")
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


@pytest.mark.skipif(
    not is_torch_npu_available() or torch.npu.device_count() < 2,
    reason="Requires at least 2 NPUs",
)
def test_distributed_ops(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29501")
    WORLD_SIZE = 2
    mp.spawn(
        _dist_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )
