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

from llamafactory.v1.accelerator.interface import DistributedInterface


def test_distributed_interface():
    DistributedInterface()
    assert DistributedInterface.get_rank() == int(os.getenv("RANK", "0"))
    assert DistributedInterface.get_world_size() == int(os.getenv("WORLD_SIZE", "1"))
    assert DistributedInterface.get_local_rank() == int(os.getenv("LOCAL_RANK", "0"))
    assert DistributedInterface.get_local_world_size() == int(os.getenv("LOCAL_WORLD_SIZE", "1"))
