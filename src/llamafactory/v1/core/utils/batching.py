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

"""Batching utils supports stateful dataloader.

1. Init stateful dataloader (tokenize)
2. Add to buffer
3. Yield batch indexes (micro batch * grad acc)
    a) non pack + non dynamic
    b) non pack + dynamic
    c) pack + non dynamic
    d) pack + dynamic
"""

from collections.abc import Iterator
from typing import Any

from torch.utils.data import default_collate
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from ...accelerator.interface import DistributedInterface
from ...config import BatchingStrategy
from ...utils import logging
from ...utils.helper import pad_and_truncate
from ...utils.types import BatchInput, ModelInput, TorchDataset
from .rendering import Renderer


logger = logging.get_logger(__name__)


def default_collate_fn(
    buffer: list[ModelInput], buffer_tokens: int, micro_batch_size: int, num_micro_batch: int, cutoff_len: int
) -> tuple[list[ModelInput], int, list[BatchInput]]:
    batch_size = micro_batch_size * num_micro_batch
    if len(buffer) < batch_size:
        return buffer, buffer_tokens, None

    samples = buffer[:batch_size]
    buffer = buffer[batch_size:]
    buffer_tokens -= sum(len(sample["input_ids"]) for sample in samples)

    batch = []
    for i in range(num_micro_batch):
        micro_batch = samples[i * micro_batch_size : (i + 1) * micro_batch_size]
        batch.append(default_collate(pad_and_truncate(micro_batch, cutoff_len)))

    return buffer, buffer_tokens, batch


class BatchGenerator(Iterator):
    def __init__(
        self,
        dataset: TorchDataset,
        renderer: Renderer,
        micro_batch_size: int = 1,
        global_batch_size: int | None = None,
        cutoff_len: int = 2048,
        batching_workers: int = 0,
        batching_strategy: BatchingStrategy = BatchingStrategy.NORMAL,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.renderer = renderer

        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.cutoff_len = cutoff_len
        self.batching_workers = batching_workers
        self.batching_strategy = batching_strategy
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        # TODO: support length and infinity

        dp_size = DistributedInterface().get_world_size("dp")

        if self.global_batch_size is None:
            self.global_batch_size = dp_size * micro_batch_size
            self.num_micro_batch = 1
        elif self.global_batch_size % (dp_size * micro_batch_size) == 0:
            self.num_micro_batch = global_batch_size // dp_size // micro_batch_size
        else:
            raise ValueError(
                "Global batch size must be divisible by DP size and micro batch size. "
                f"Got {global_batch_size} % ({dp_size} * {micro_batch_size}) != 0."
            )

        if not self.drop_last:
            raise ValueError("Drop last must be True.")

        self._init_data_provider()

        self._is_resuming: bool = False
        self._data_iter = iter(self._data_provider)
        self._buffer: list[ModelInput] = []
        self._buffer_tokens: int = 0
        self._max_buffer_tokens: int = self.micro_batch_size * self.num_micro_batch * self.cutoff_len

        logger.info_rank0(
            f"Init unified data loader with global batch size {self.global_batch_size}, "
            f"micro batch size {self.micro_batch_size}, "
            f"num micro batch {self.num_micro_batch}, "
            f"cutoff len {self.cutoff_len}, "
            f"batching workers {self.batching_workers}, "
            f"batching strategy {self.batching_strategy}."
        )

    def _init_data_provider(self) -> None:
        if len(self.dataset) != -1:
            sampler = StatefulDistributedSampler(
                self.dataset,
                num_replicas=DistributedInterface().get_world_size("dp"),
                rank=DistributedInterface().get_rank("dp"),
                shuffle=True,
                seed=0,
                drop_last=self.drop_last,
            )
        else:
            raise NotImplementedError("Iterable dataset is not supported yet.")

        self._data_provider = StatefulDataLoader(
            self.dataset,
            batch_size=self.micro_batch_size * self.num_micro_batch,
            sampler=sampler,
            num_workers=self.batching_workers,
            collate_fn=self.renderer.process_samples,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        if self.batching_strategy == BatchingStrategy.NORMAL:
            self._length = len(self._data_provider)
        else:
            from ...plugins.trainer_plugins.batching import BatchingPlugin

            self._length = BatchingPlugin(self.batching_strategy).compute_length()
            raise NotImplementedError("Batching strategy other than NORMAL is not supported yet.")

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        if not self._is_resuming:
            self._buffer.clear()
            self._buffer_tokens = 0

        self._data_iter = iter(self._data_provider)
        self._is_resuming = False
        return self

    def __next__(self):
        batch = self._next_batch()
        if batch is None:
            raise StopIteration

        return batch

    def _next_batch(self) -> list[BatchInput] | None:
        while self._buffer_tokens < self._max_buffer_tokens:
            try:
                samples: list[ModelInput] = next(self._data_iter)
            except StopIteration:
                break

            num_tokens = sum(len(sample["input_ids"]) for sample in samples)
            self._buffer.extend(samples)
            self._buffer_tokens += num_tokens

        return self._build_batch()

    def _build_batch(self) -> list[BatchInput] | None:
        if self.batching_strategy == BatchingStrategy.NORMAL:
            self._buffer, self._buffer_tokens, batch = default_collate_fn(
                self._buffer, self._buffer_tokens, self.micro_batch_size, self.num_micro_batch, self.cutoff_len
            )
            return batch
        else:
            from ...plugins.trainer_plugins.batching import BatchingPlugin

            self._buffer, self._buffer_tokens, batch = BatchingPlugin(self.batching_strategy)(
                self._buffer, self._buffer_tokens, self.micro_batch_size, self.num_micro_batch, self.cutoff_len
            )
            return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "buffer": self._buffer,
            "buffer_tokens": self._buffer_tokens,
            "data_provider": self._data_provider.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._buffer = state["buffer"]
        self._buffer_tokens = state["buffer_tokens"]
        self._data_provider.load_state_dict(state["data_provider"])
        self._is_resuming = True

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._data_provider.sampler, "set_epoch"):
            self._data_provider.sampler.set_epoch(epoch)


if __name__ == "__main__":
    """
    python -m llamafactory.v1.core.utils.batching \
        --model llamafactory/tiny-random-qwen2.5 \
        --dataset data/v1_sft_demo.yaml \
        --micro_batch_size 2 \
        --global_batch_size 4 \
        --batching_workers 0
    """
    from ...config.arg_parser import get_args
    from ..data_engine import DataEngine
    from ..model_engine import ModelEngine

    data_args, model_args, training_args, _ = get_args()
    data_engine = DataEngine(data_args=data_args)
    model_engine = ModelEngine(model_args=model_args)
    batch_generator = BatchGenerator(
        data_engine,
        model_engine.renderer,
        micro_batch_size=training_args.micro_batch_size,
        global_batch_size=training_args.global_batch_size,
        cutoff_len=training_args.cutoff_len,
        batching_workers=training_args.batching_workers,
        batching_strategy=training_args.batching_strategy,
    )
    for batch in batch_generator:
        print(batch)
        print(len(batch))
        print(batch[0]["input_ids"].shape)
        break
