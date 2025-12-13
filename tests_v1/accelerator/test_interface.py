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

from llamafactory.v1.accelerator.interface import DistributedInterface, DistributedStrategy


def test_distributed_interface():
    DistributedInterface(DistributedStrategy())
    assert DistributedInterface.get_rank() == int(os.getenv("RANK", "0"))
    assert DistributedInterface.get_world_size() == int(os.getenv("WORLD_SIZE", "1"))
    assert DistributedInterface.get_local_rank() == int(os.getenv("LOCAL_RANK", "0"))
    assert DistributedInterface.get_local_world_size() == int(os.getenv("LOCAL_WORLD_SIZE", "1"))


def test_npu_all_reduce():
    import numpy as np
    import torch
    import torch.distributed as dist

    from llamafactory.v1.accelerator.helper import ReduceOp, is_torch_npu_available

    if not is_torch_npu_available():
        return

    DistributedInterface(DistributedStrategy())
    local_rank = DistributedInterface.get_local_rank()
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("hccl")

    rank = DistributedInterface.get_rank()
    print(f"[Rank {rank}] Device: {torch.npu.current_device()}")

    # Case 1: Tensor
    input_tensor = torch.tensor([2.0 * (rank + 1)]).to(f"npu:{local_rank}")
    res_tensor = DistributedInterface.all_reduce(input_tensor, op=ReduceOp.MEAN)
    assert res_tensor.item() == 3.0
    if rank == 0:
        print("Case 1 Passed")

    # Case 2: Numpy
    input_np = np.array([1.0, 2.0]).astype(np.float32)
    res_np = DistributedInterface.all_reduce(input_np, op=ReduceOp.SUM)
    assert np.allclose(res_np, np.array([2.0, 4.0]))
    if rank == 0:
        print("Case 2 Passed")

    # Case 3: Scalar
    input_scalar = 10.0 * (rank + 1)
    res_scalar = DistributedInterface.all_reduce(input_scalar, op=ReduceOp.MAX)
    assert res_scalar == 20.0
    if rank == 0:
        print("Case 3 Passed")

    dist.destroy_process_group()
