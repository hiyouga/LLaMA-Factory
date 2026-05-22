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

from typing import Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup


_CONTEXT_PARALLEL_GROUP: Optional[ProcessGroup] = None


def set_context_parallel_group(group: ProcessGroup) -> None:
    """Set context parallel (sequence parallel) process group."""
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = group


def get_context_parallel_group() -> Optional[ProcessGroup]:
    """Get context parallel process group."""
    global _CONTEXT_PARALLEL_GROUP
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_world_size(group: ProcessGroup = None) -> int:
    """Get context parallel world size."""
    group = get_context_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


def get_context_parallel_rank(group: ProcessGroup = None) -> int:
    """Get context parallel rank."""
    group = get_context_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0
