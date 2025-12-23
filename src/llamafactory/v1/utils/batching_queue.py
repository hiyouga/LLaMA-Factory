# Copyright 2025 Bytedance Ltd. and the LlamaFactory team.
#
# This code is inspired by the Bytedance's VeOmni library.
# https://github.com/ByteDance-Seed/VeOmni/blob/v0.1.4/veomni/data/dynamic_batching.py
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

from abc import ABC, abstractmethod


class DynamicBatchSizeBuffer:
    """A buffer to store samples for dynamic batch size."""

    def __init__(self):
        self._buffer: list[dict[str, any]] = []
        self._buffer_sample_lengths: list[int] = []
        self._deleted_indices: set[int] = set()
        self._current_index: int = 0
        self._total_token_count: int = 0

    def append(self, item: dict[str, any]) -> None:
        """Append a sample to the buffer.

        Args:
            item: A sample to append to the buffer.
                The sample should be a dict with the following keys:
                    - input_ids: torch.Tensor of shape (seq_len, )
                    - attention_mask: torch.Tensor of shape (seq_len, )
        """
        self._buffer.append(item)
        sample_length = int(item["attention_mask"].sum().item())
        self._buffer_sample_lengths.append(sample_length)
        self._total_token_count += sample_length

    def get_samples(self, max_tokens_per_iteration: int, force: bool = True) -> list[dict[str, any]]:
        """Get samples from the buffer that fit within the token budget.

        Args:
            max_tokens_per_iteration: Maximum number of tokens to retrieve.
            force: If True, the first available sample will be returned even
                if it exceeds the token budget.

        Returns:
            A list of samples that fit within the token budget.

        Raises:
            AssertionError: If no samples are found (should not happen in normal operation).
        """
        cum_seq_len = 0
        samples = []

        while self._current_index < len(self._buffer) and cum_seq_len < max_tokens_per_iteration:
            if self._current_index in self._deleted_indices:
                self._current_index += 1
                continue

            seq_len = self._buffer_sample_lengths[self._current_index]
            remaining_tokens = max_tokens_per_iteration - cum_seq_len

            # Check if we can add this sample
            can_add = (force and cum_seq_len == 0) or (seq_len <= remaining_tokens)

            if can_add:
                cum_seq_len += seq_len
                samples.append(self._buffer[self._current_index])
                self._deleted_indices.add(self._current_index)

            self._current_index += 1

        assert len(samples) > 0, "No samples found in buffer"
        return samples

    def __len__(self) -> int:
        """Return the number of samples in the buffer."""
        return len(self._buffer)

    @property
    def total_token_count(self) -> int:
        """Return the total number of tokens in the buffer."""
        return self._total_token_count

    def flush(self) -> None:
        tokens_to_remove = sum(self._buffer_sample_lengths[idx] for idx in self._deleted_indices)
        self._total_token_count -= tokens_to_remove

        buffer_length = len(self._buffer)
        self._buffer = [self._buffer[idx] for idx in range(buffer_length) if idx not in self._deleted_indices]
        self._buffer_sample_lengths = [
            self._buffer_sample_lengths[idx] for idx in range(buffer_length) if idx not in self._deleted_indices
        ]

        self._current_index = 0
        self._deleted_indices.clear()


class BaseBatchingQueue(ABC):
    """Base class for batching queue."""

    @abstractmethod
    def is_full_filled(self) -> bool:
        raise NotImplementedError("Subclasses must implement `is_full_filled`")

    @abstractmethod
    def put_item(self, item: dict[str, any]) -> None:
        raise NotImplementedError("Subclasses must implement `put_item`")

    @abstractmethod
    def get_micro_batch(self, step: int) -> list[dict[str, any]]:
        raise NotImplementedError("Subclasses must implement `get_micro_batch`")

    @abstractmethod
    def empty(self) -> bool:
        raise NotImplementedError("Subclasses must implement `empty`")


class IdentityPacker:
    def __init__(self, token_micro_bsz, bsz_warmup_steps, bsz_warmup_init_mbtoken):
        self.token_micro_bsz = token_micro_bsz
        self.bsz_warmup_steps = bsz_warmup_steps
        self.bsz_warmup_init_mbtoken = bsz_warmup_init_mbtoken

    def __call__(self, samples):
        return samples

    def get_token_num_to_request(self, cur_step, warmup):
        return (
            (self.token_micro_bsz - self.bsz_warmup_init_mbtoken) * cur_step // self.bsz_warmup_steps
            + self.bsz_warmup_init_mbtoken
            if warmup
            else self.token_micro_bsz
        )


class TextBatchingQueue(BaseBatchingQueue):
    """Batching text queue for text data."""

    def __init__(
        self,
        token_micro_bsz,
        buffer_size: int = 500,
        bsz_warmup_steps: int = -1,
        bsz_warmup_init_mbtoken: int = 200,
    ) -> None:
        super().__init__()
        self._step = 0
        self.token_micro_bsz = token_micro_bsz
        self.bsz_warmup_steps = bsz_warmup_steps
        self.buffer_size = buffer_size  # minimum samples in buffer
        self.buffer = DynamicBatchSizeBuffer()
        self.bsz_warmup_init_mbtoken = bsz_warmup_init_mbtoken  # training warmup args
        assert self.bsz_warmup_init_mbtoken >= 0

        self.packer = IdentityPacker(
            token_micro_bsz=token_micro_bsz,
            bsz_warmup_steps=bsz_warmup_steps,
            bsz_warmup_init_mbtoken=bsz_warmup_init_mbtoken,
        )

    def is_full_filled(self) -> bool:
        return len(self.buffer) >= self.buffer_size and self.buffer.total_token_count >= self.token_micro_bsz

    def put_item(self, item: dict[str, any]):
        if len(item["input_ids"]) == 1:
            print("WARNING: EMPTY STRING.")
            return
        self.buffer.append(item)

    def get_token_num_to_request(self):
        if self.packer is not None:
            warmup = self._step <= self.bsz_warmup_steps and self.bsz_warmup_steps > 0
            return self.packer.get_token_num_to_request(self._step, warmup=warmup)
        else:
            return self.get_cur_token_micro_bsz()

    def get_cur_token_micro_bsz(self):
        warmup = self._step <= self.bsz_warmup_steps and self.bsz_warmup_steps > 0
        if warmup:
            return (
                self.token_micro_bsz - self.bsz_warmup_init_mbtoken
            ) * self._step // self.bsz_warmup_steps + self.bsz_warmup_init_mbtoken
        else:
            return self.token_micro_bsz

    def get_micro_batch(self, step) -> any:
        """Get a micro batch from the buffer according to the current step.

        Args:
            step: the current step.

        Returns:
            data: a list of samples.
        """
        self._step = step
        n_token_per_iter = self.get_token_num_to_request()
        cur_token_micro_bsz = self.get_cur_token_micro_bsz()
        assert cur_token_micro_bsz % n_token_per_iter == 0, (
            "The token num to get for each request should be divisible by token micro bsz."
        )
        n_iter = int(cur_token_micro_bsz // n_token_per_iter)
        data = []
        for _ in range(n_iter):
            samples = self.buffer.get_samples(n_token_per_iter)
            if self.packer:
                samples = self.packer(samples)  # maybe packed into one sample, but wrapped in list.
            data.extend(samples)
        self.buffer.flush()  # remove the selected samples.
        return data

    def empty(self) -> bool:
        return len(self.buffer) == 0
