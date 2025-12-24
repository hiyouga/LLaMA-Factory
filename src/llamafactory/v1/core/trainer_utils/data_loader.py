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


import copy
import sys
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Optional

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from ...utils.batching_queue import BaseBatchingQueue
from ...utils.logging import get_logger
from ...utils.types import Processor, TorchDataset
from .data_collator import DataCollator


logger = get_logger(__name__)


# base dataloader
class DistributedDataloader(StatefulDataLoader):
    """Base Distributed DataLoader."""

    dataset: "TorchDataset"
    sampler: "StatefulDistributedSampler"

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


@dataclass
class BaseDataLoader:
    """Default DataLoader."""

    processor: Processor

    def __init__(self, dataset: TorchDataset) -> None:
        self.dataset = dataset
        # guidlines: fetch until get fixed batchsize.
        # save state_dict for buffer.
        # resume with state

        # 1. Init stateful dataloader (tokenize)
        # 2. Add to buffer (2 * max seq len per device)
        # 3. Yield batch indexes (micro batch * grad acc)
        #    a ) non pack + non dynamic
        #    b ) non pack + dynamic
        #    c ) pack + non dynamic
        #    d ) pack + dynamic

    def init_dataloader(self) -> None:
        ### init dataloader
        pass

    def __iter__(self) -> Iterator:
        pass

    def __next__(self) -> any:
        pass


@dataclass
class DataLoader:
    """Default DataLoader."""

    processor: "Processor"
    dataloader: "DistributedDataloader"
    batching_queue: "BaseBatchingQueue"
    collate_fn: "DataCollator"
    num_micro_batch: int = 1
    length: int = 0
    drop_last: bool = True

    def __init__(
        self,
        dataloader: any,
        collate_fn: "DataCollator",
        num_micro_batch: int = 1,
        length: int = 0,
        drop_last: bool = True,
        batching_queue: Optional["BaseBatchingQueue"] = None,
    ) -> None:
        self.batching_queue = batching_queue
        self.num_micro_batch = num_micro_batch
        self.step = 0
        self._collate_fn = collate_fn
        self._dataloader = dataloader
        self._drop_last = drop_last
        self._data_iter: Iterator
        self._resume = False
        self._batch_data_iter: Generator

        if length > 0:
            self._length = length
        elif length == -1:
            self._length = sys.maxsize
        else:
            self._length = len(self._dataloader)

    def __len__(self):
        return self._length

    def __iter__(self) -> Iterator:
        if not self._resume:
            self.step = 0
            self._data_iter = iter(self._dataloader)
            self._batch_data_iter = self.batch_data_generator()
        self._resume = False
        return self

    def __next__(self):
        return next(self._batch_data_iter)  # FIXME maybe we can move origin_batch_data_generator to here

    def origin_batch_data_generator(self):
        """Standard pass-through generator if do not use batching queue."""
        while True:
            if self._length > 0 and self.step >= self._length:
                return

            try:
                batch = []
                data = next(self._data_iter)
                # split data into micro batches
                for i in range(0, len(data), self.num_micro_batch):
                    micro_batch = data[i : i + self.num_micro_batch]
                    if self._collate_fn:
                        micro_batch = self._collate_fn(micro_batch)
                    batch.append(micro_batch)
                yield batch
                self.step += 1
            except StopIteration:
                if self.step < self._length:
                    # Restart iterator to fill the requested length
                    self._data_iter = iter(self._dataloader)
                    try:
                        batch = []
                        data = next(self._data_iter)
                        for i in range(0, len(data), self.num_micro_batch):
                            micro_batch = data[i : i + self.num_micro_batch]
                            if self._collate_fn:
                                micro_batch = self._collate_fn(micro_batch)
                            batch.append(micro_batch)
                        yield batch
                        self.step += 1
                    except StopIteration:
                        return
                else:
                    return
            except Exception as e:
                logger.error(f"DataLoader origin_batch_data_generator exception: {e}")
                raise

    def batch_data_generator(self):
        if self.batching_queue is None:
            yield from self.origin_batch_data_generator()
            return

        batch = []

        while True:
            if self._length and self.step >= self._length:
                return

            if self.batching_queue.is_full_filled():
                micro_batch = self.batching_queue.get_micro_batch(self.step)
                if self._collate_fn:
                    micro_batch = self._collate_fn(micro_batch)
                batch.append(micro_batch)
                if len(batch) == self.num_micro_batch:
                    yield batch
                    self.step += 1
                    batch = []

            try:
                processing_item = next(self._data_iter)
            except Exception as e:
                if isinstance(e, StopIteration):
                    if self.step < self._length:
                        # call iter until reach length
                        self._data_iter = iter(self._dataloader)
                        processing_item = next(self._data_iter)
                    elif not self._drop_last and not self.batching_queue.empty():
                        while not self.batching_queue.empty():
                            micro_batch = self.batching_queue.get_micro_batch(self.step)
                            if self._collate_fn:
                                micro_batch = self._collate_fn(micro_batch)
                            batch.append(micro_batch)
                            if len(batch) == self.num_micro_batch:
                                yield batch
                                self.step += 1
                                batch = []

                        while len(batch) < self.num_micro_batch:
                            padding_batch = copy.deepcopy(micro_batch)
                            padding_batch["is_padded"] = True
                            batch.append(padding_batch)
                        yield batch
                        self.step += 1
                        return
                    else:
                        return
                else:
                    logger.error(f"DataLoader iter data exception: {e}")
                    raise

            # put processing_item to buffer
            if isinstance(processing_item, dict):
                processing_item = [processing_item]

            for item in processing_item:
                self.batching_queue.put_item(item)

    def state_dict(self):
        # save state
        state = self.__dict__.copy()
        # remove internal fields
        for k in list(state.keys()):
            if k.startswith("_"):
                del state[k]

        # save dataloader state
        if hasattr(self._dataloader, "state_dict"):
            state["dataloader_state"] = self._dataloader.state_dict()
        elif hasattr(self._dataloader, "__getstate__"):
            state["dataloader_state"] = self._dataloader.__getstate__()

        batching_strategy = getattr(self, "batching_strategy", None)
        if batching_strategy and hasattr(batching_strategy, "state_dict"):
            state["batching_strategy_state"] = batching_strategy.state_dict()
            if "batching_strategy" in state:
                del state["batching_strategy"]

        return copy.deepcopy(state)

    def load_state_dict(self, state: dict[str, any]):
        if state["num_micro_batch"] != self.num_micro_batch:
            logger.warning(
                f"num_micro_batch changed: [ {state['num_micro_batch']} -> {self.num_micro_batch} ], will clear prefetch buffer"
            )
            del state["num_micro_batch"]
        self.__dict__.update(state)
        self._resume = True

        if hasattr(self._dataloader, "load_state_dict"):
            self._dataloader.load_state_dict(state["dataloader_state"])
        elif hasattr(self._dataloader, "__getstate__"):
            self._dataloader.__setstate__(state["dataloader_state"])

        if "batching_strategy_state" in state:
            batching_strategy = getattr(self, "batching_strategy", None)
            if batching_strategy:
                batching_strategy.load_state_dict(state["batching_strategy_state"])
            del state["batching_strategy_state"]

        self._data_iter = iter(self._dataloader)
        self._batch_data_iter = self.batch_data_generator()

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._dataloader, "set_epoch"):
            self._dataloader.set_epoch(epoch)
