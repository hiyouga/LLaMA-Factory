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


import torch

from ...accelerator.helper import DeviceType
from ...accelerator.interface import DistributedInterface
from ...utils.plugin import BasePlugin


class InitPlugin(BasePlugin):
    def __call__(self) -> torch.device:
        return super().__call__()


@InitPlugin("init_on_meta").register()
def init_on_meta() -> torch.device:
    return torch.device(DeviceType.META.value)


@InitPlugin("init_on_rank0").register()
def init_on_rank0() -> torch.device:
    if DistributedInterface().get_rank() == 0:
        return torch.device(DeviceType.CPU.value)
    else:
        return torch.device(DeviceType.META.value)


@InitPlugin("init_on_default").register()
def init_on_default() -> torch.device:
    return DistributedInterface().current_device
