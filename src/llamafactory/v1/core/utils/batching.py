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

from ...accelerator.interface import Dim, DistributedInterface
from ...config import BatchingStrategy
from ...utils import logging
from ...utils.helper import pad_and_truncate
from ...utils.objects import StatefulBuffer
from ...utils.types import BatchInfo, BatchInput, ModelInput, TorchDataset
from .rendering import Renderer


logger = logging.get_logger(__name__)


def default_collate_fn(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
    micro_batch_size = batch_info["micro_batch_size"]
    num_micro_batch = batch_info["num_micro_batch"]
    cutoff_len = batch_info["cutoff_len"]
    batch_size = micro_batch_size * num_micro_batch
    if len(buffer) < batch_size:
        return None

    samples = buffer.get(batch_size)
    batch = []
    for i in range(num_micro_batch):
        micro_batch = samples[i * micro_batch_size : (i + 1) * micro_batch_size]
        batch.append(default_collate(pad_and_truncate(micro_batch, cutoff_len)))

    return batch


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
        dp_size = DistributedInterface().get_world_size(Dim.DP)

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
        self._buffer = StatefulBuffer()

        self._batch_info: BatchInfo = {
            "micro_batch_size": self.micro_batch_size,
            "num_micro_batch": self.num_micro_batch,
            "cutoff_len": self.cutoff_len,
            "data_iter": self._data_iter,
        }

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
                num_replicas=DistributedInterface().get_world_size(Dim.DP),
                rank=DistributedInterface().get_rank(Dim.DP),
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
            pin_memory_device=DistributedInterface().current_device.type,
            drop_last=self.drop_last,
        )
        if self.batching_strategy == BatchingStrategy.NORMAL:
            self._length = len(self._data_provider)
        else:
            from ...plugins.trainer_plugins.batching import BatchingPlugin

            self._length = BatchingPlugin(self.batching_strategy).compute_length(self._data_provider)
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
        self._fill_buffer()
        batch = self._generate_batch()
        if batch is None:
            raise StopIteration

        return batch

    def _fill_buffer(self) -> None:
        if self.batching_strategy == BatchingStrategy.NORMAL:
            while len(self._buffer) < self.micro_batch_size * self.num_micro_batch:
                try:
                    samples: list[ModelInput] = next(self._data_iter)
                except StopIteration:
                    break

                self._buffer.put(samples)
        else:
            from ...plugins.trainer_plugins.batching import BatchingPlugin

            BatchingPlugin(self.batching_strategy).fill_buffer(self._buffer, self._batch_info)

    def _generate_batch(self) -> list[BatchInput] | None:
        if self.batching_strategy == BatchingStrategy.NORMAL:
            return default_collate_fn(self._buffer, self._batch_info)
        else:
            from ...plugins.trainer_plugins.batching import BatchingPlugin

            return BatchingPlugin(self.batching_strategy).generate_batch(self._buffer, self._batch_info)

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
        --train_dataset data/v1_sft_demo.yaml \
        --micro_batch_size 2 \
        --global_batch_size 4 \
        --batching_workers 0
    """
    from ...config.arg_parser import get_args
    from ..data_engine import DataEngine
    from ..model_engine import ModelEngine

    model_args, data_args, training_args, _ = get_args()
    data_engine = DataEngine(data_args.train_dataset)
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
