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

from ...utils.objects import StatefulBuffer
from ...utils.plugin import BasePlugin
from ...utils.types import BatchInfo, BatchInput


class BatchingPlugin(BasePlugin):
    def compute_length(self, batch_info: BatchInfo) -> int:
        raise NotImplementedError()

    def fill_buffer(self, buffer: StatefulBuffer, batch_info: BatchInfo) -> None:
        raise NotImplementedError()

    def generate_batch(self, buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
        raise NotImplementedError()
