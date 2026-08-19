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


@dataclass(frozen=True)
class ChunkSizePolicy:
    """Resolve a fixed chunk size or a batch-aware token budget."""

    fixed_chunk_size: int | None = None
    token_budget: int | None = None

    def __post_init__(self) -> None:
        if (self.fixed_chunk_size is None) == (self.token_budget is None):
            raise ValueError("Exactly one of `chunk_loss_size` and `chunk_loss_token_budget` must be configured.")
        if self.fixed_chunk_size is not None and self.fixed_chunk_size <= 0:
            raise ValueError("`chunk_loss_size` must be positive when chunk loss is enabled.")
        if self.token_budget is not None and self.token_budget <= 0:
            raise ValueError("`chunk_loss_token_budget` must be positive when chunk loss is enabled.")

    def resolve(self, batch_size: int, sequence_length: int) -> int:
        if self.fixed_chunk_size is not None:
            return min(self.fixed_chunk_size, sequence_length)

        assert self.token_budget is not None
        per_sequence_budget = max(self.token_budget // batch_size, 1)
        chunk_size = 1 << (per_sequence_budget.bit_length() - 1)
        return min(chunk_size, sequence_length)
