# test allreduce on npu
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llamafactory.v1.accelerator.helper import ReduceOp, all_reduce, is_torch_npu_available


os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

WORLD_SIZE = 2


def _dist_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

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
    if rank == 0:
        print("Case 1 Passed")

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
def test_distributed_ops():
    mp.spawn(
        _dist_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )
