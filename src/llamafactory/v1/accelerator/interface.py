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

from dataclasses import dataclass
from typing import Any, Optional

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from ..utils.types import TensorLike
from .helper import ReduceOp, all_reduce, get_current_accelerator, get_rank, get_world_size, is_distributed


@dataclass
class DistributedStrategy:
    """Distributed strategy."""

    dp_size: Optional[int] = None
    tp_size: int = 1

    def __post_init__(self) -> None:
        if not is_distributed():
            self.dp_size = 1
        elif self.dp_size is None:
            self.dp_size = get_world_size() // self.tp_size
        elif self.dp_size * self.tp_size != get_world_size():
            raise ValueError(
                f"dp_size * tp_size must equal to world_size, "
                f"got {self.dp_size} * {self.tp_size} != {get_world_size()}."
            )

    @property
    def mesh_shape(self) -> tuple[int, int]:
        """Mesh shape."""
        return (self.dp_size, self.tp_size)

    @property
    def mesh_dim_names(self) -> tuple[str, str]:
        """Mesh dimension names."""
        return ("dp", "tp")


class DistributedInterface:
    """Distributed interface."""

    _instance: Optional["DistributedInterface"] = None
    _initialized: bool = False

    is_distributed = is_distributed()
    """Check if distributed environment is available."""
    rank = get_rank()
    """Global rank."""
    world_size = get_world_size()
    """Global world size."""
    device_mesh: Optional[DeviceMesh] = None
    """Device mesh."""
    current_accelerator = get_current_accelerator()
    """Current accelerator."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "DistributedInterface":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self, strategy: DistributedStrategy) -> None:
        if self._initialized:
            return

        self.strategy = strategy
        if self.is_distributed:
            self.device_mesh = init_device_mesh(
                device_type=self.current_accelerator.type,
                mesh_shape=strategy.mesh_shape,
                mesh_dim_names=strategy.mesh_dim_names,
            )
        else:
            self.device_mesh = None

        self._initialized = True

    def __str__(self) -> str:
        return (
            f"DistributedInterface(strategy={self.strategy}), is_distributed={self.is_distributed}, "
            f"rank={self.rank}, world_size={self.world_size}, "
            f"device_mesh={self.device_mesh}, current_accelerator={self.current_accelerator}"
        )

    def dp_rank(self) -> int:
        """Data parallel rank."""
        if self.device_mesh is None:
            return 0

        return self.device_mesh["dp"].get_rank()

    def dp_size(self) -> int:
        """Data parallel size."""
        if self.device_mesh is None:
            return 1

        return self.device_mesh["dp"].size()

    def all_reduce_over_dp(self, data: TensorLike, op: ReduceOp = ReduceOp.MEAN) -> TensorLike:
        """All reduce tensor."""
        if self.device_mesh is None:
            return data

        return all_reduce(data, op, self.device_mesh["dp"].get_group())


if __name__ == "__main__":
    print(DistributedInterface(DistributedStrategy()))
