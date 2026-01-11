# Copyright 2025 Optuna, HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/utils/logging.py
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

from .types import ModelInput


class StatefulBuffer:
    """A buffer that stores model inputs."""

    def __init__(self, max_buffer_size: int = 1_000_000_000) -> None:
        self._buffer: list[ModelInput] = []
        self._buffer_size: int = 0
        self._max_buffer_size: int = max_buffer_size

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def size(self) -> int:
        return self._buffer_size

    def put(self, samples: list[ModelInput]) -> None:
        """Add samples to the buffer."""
        num_tokens = sum(len(sample["input_ids"]) for sample in samples)
        if self._buffer_size + num_tokens > self._max_buffer_size:
            raise ValueError(f"Buffer size exceeds max buffer size {self._max_buffer_size}.")

        self._buffer.extend(samples)
        self._buffer_size += num_tokens

    def get(self, value: int) -> list[ModelInput]:
        """Get samples from the buffer and remove them."""
        samples = self._buffer[:value]
        self._buffer_size -= sum(len(sample["input_ids"]) for sample in samples)
        del self._buffer[:value]
        return samples

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer = []
        self._buffer_size = 0

    def state_dict(self) -> dict:
        """Returns the state of the buffer."""
        return {
            "buffer": self._buffer,
            "buffer_size": self._buffer_size,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the state into the buffer."""
        self._buffer = state_dict["buffer"]
        self._buffer_size = state_dict["buffer_size"]
