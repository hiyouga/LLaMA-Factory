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

import torch
from torch.utils.data import default_collate
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from ...accelerator.helper import ReduceOp
from ...accelerator.interface import Dim, DistributedInterface
from ...config import BatchingStrategy
from ...utils import logging
from ...utils.constants import IGNORE_INDEX
from ...utils.helper import is_tokenizer
from ...utils.objects import StatefulBuffer
from ...utils.types import BatchInfo, BatchInput, ModelInput, TorchDataset
from .rendering import _MULTIMODAL_PASSTHROUGH_KEYS, Renderer, _align_multimodal_on_truncation, pad_and_truncate


logger = logging.get_logger(__name__)

__all__ = ["BatchGenerator"]


# (modality, pixel key, grid key, mm_token_type_ids marker) for vision-tower alignment.
_ALIGN_MODALITIES = (
    ("image", "pixel_values", "image_grid_thw", 1),
    ("video", "pixel_values_videos", "video_grid_thw", 2),
)


def _inject_dummy_media(micro_batch: list[ModelInput], fragment: dict, cutoff_len: int) -> None:
    """Append a zero-loss dummy media fragment to one sample of ``micro_batch`` in place.

    The fragment is added at the end of the chosen sample so causal attention keeps every real
    token's logits unchanged; its placeholder tokens carry ``IGNORE_INDEX`` labels and zero loss
    weight, so the dummy contributes nothing to the loss while forcing the vision tower to run.
    """
    frag_ids = fragment["input_ids"]
    frag_len = len(frag_ids)
    frag_mm = fragment["mm_token_type_ids"]

    # Prefer a pure-text sample (no media to orphan when trimming); else the shortest one.
    def _is_text_only(s: ModelInput) -> bool:
        return not any(k in s for k in _MULTIMODAL_PASSTHROUGH_KEYS)

    text_only = [j for j, s in enumerate(micro_batch) if _is_text_only(s)]
    candidates = text_only if text_only else range(len(micro_batch))
    idx = min(candidates, key=lambda j: len(micro_batch[j]["input_ids"]))

    s = dict(micro_batch[idx])
    n = len(s["input_ids"])

    # Reserve room so the dummy survives cutoff truncation; trim the real tail if needed.
    keep = max(0, cutoff_len - frag_len)
    if n > keep:
        for key in ("input_ids", "attention_mask", "labels", "loss_weights", "position_ids", "mm_token_type_ids"):
            if key in s:
                s[key] = list(s[key])[:keep]
        # Trimming may orphan this sample's own media; realign before appending the dummy.
        if any(k in s for k in _MULTIMODAL_PASSTHROUGH_KEYS):
            s = dict(_align_multimodal_on_truncation(s, keep))
        n = len(s["input_ids"])

    s["input_ids"] = list(s["input_ids"]) + list(frag_ids)
    s["attention_mask"] = list(s["attention_mask"]) + [1] * frag_len
    s["labels"] = list(s["labels"]) + [IGNORE_INDEX] * frag_len
    s["loss_weights"] = list(s["loss_weights"]) + [0.0] * frag_len
    s["mm_token_type_ids"] = list(s.get("mm_token_type_ids", [0] * n)) + list(frag_mm)
    if "position_ids" in s:
        base_pos = list(s["position_ids"])
        last = base_pos[-1] if base_pos else 0
        s["position_ids"] = base_pos + list(range(last + 1, last + 1 + frag_len))

    for key, value in fragment.items():
        if key in ("input_ids", "mm_token_type_ids"):
            continue
        s[key] = torch.cat([s[key], value], dim=0) if key in s else value

    micro_batch[idx] = s


def default_collate_fn(
    buffer: StatefulBuffer, batch_info: BatchInfo, renderer: Renderer | None = None
) -> list[BatchInput] | None:
    micro_batch_size = batch_info["micro_batch_size"]
    num_micro_batch = batch_info["num_micro_batch"]
    cutoff_len = batch_info["cutoff_len"]
    batch_size = micro_batch_size * num_micro_batch
    if len(buffer) < batch_size:
        return None

    samples = buffer.get(batch_size)
    micro_batches = [samples[i * micro_batch_size : (i + 1) * micro_batch_size] for i in range(num_micro_batch)]

    # Vision-tower alignment: a micro batch with media makes the (FSDP-sharded) vision blocks
    # fire collectives that a media-less micro batch on a sibling DP rank never issues -> NCCL
    # hang. Negotiate, per micro batch, which modalities are present *anywhere* in the DP group
    # (all_reduce MAX over local presence), then inject a dummy into the micro batches that lack
    # a globally-present modality so every rank invokes the vision tower the same number of times.
    # NOTE: this all_reduce must run the same number of times on every rank; it is reached only
    # on the full-batch path (the `len(buffer) < batch_size` early return above is shared by all
    # ranks thanks to drop_last + equal shards), so the count stays aligned.
    if renderer is not None and not is_tokenizer(renderer.processor):
        present = torch.zeros((num_micro_batch, len(_ALIGN_MODALITIES)), dtype=torch.int64)
        for i, mb in enumerate(micro_batches):
            for m, (_, pixel_key, _, _) in enumerate(_ALIGN_MODALITIES):
                present[i, m] = int(any(pixel_key in s for s in mb))

        present = DistributedInterface().all_reduce(present, op=ReduceOp.MAX, dim=Dim.DP)

        for i, mb in enumerate(micro_batches):
            for m, (modality, pixel_key, _, _) in enumerate(_ALIGN_MODALITIES):
                if present[i, m] and not any(pixel_key in s for s in mb):
                    _inject_dummy_media(mb, renderer.get_dummy_media_fragment(modality), cutoff_len)

    batch = []
    for micro_batch in micro_batches:
        padded = pad_and_truncate(micro_batch, cutoff_len)

        standard_samples = [{k: v for k, v in s.items() if k not in _MULTIMODAL_PASSTHROUGH_KEYS} for s in padded]
        collated = default_collate(standard_samples)

        for key in _MULTIMODAL_PASSTHROUGH_KEYS:
            tensors = [s[key] for s in padded if key in s]
            if tensors:
                collated[key] = torch.cat(tensors, dim=0)

        batch.append(collated)

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
        seed: int = 42,
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
        self.seed = seed
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

        self._batch_info: BatchInfo = {
            "micro_batch_size": self.micro_batch_size,
            "num_micro_batch": self.num_micro_batch,
            "cutoff_len": self.cutoff_len,
        }

        self._init_data_provider()

        self._is_resuming: bool = False
        self._data_iter = iter(self._data_provider)
        self._buffer = StatefulBuffer()

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
                seed=self.seed,
                drop_last=self.drop_last,
            )
        else:
            raise NotImplementedError("Iterable dataset is not supported yet.")

        if self.batching_strategy == BatchingStrategy.NORMAL:
            batch_size = self.micro_batch_size * self.num_micro_batch
        else:
            from ...plugins.trainer_plugins.batching import BatchingPlugin

            batch_size = BatchingPlugin(self.batching_strategy).get_data_provider_batch_size(self._batch_info)

        generator_seed = torch.Generator()
        generator_seed.manual_seed(self.seed)

        self._data_provider = StatefulDataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.batching_workers,
            collate_fn=self.renderer.process_samples,
            pin_memory=self.pin_memory,
            pin_memory_device=DistributedInterface().current_device.type,
            drop_last=self.drop_last,
            generator=generator_seed,
        )
        if self.batching_strategy == BatchingStrategy.NORMAL:
            self._length = len(self._data_provider)
        else:
            from ...plugins.trainer_plugins.batching import BatchingPlugin

            self._length = BatchingPlugin(self.batching_strategy).compute_length(self._data_provider, self._batch_info)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        if not self._is_resuming:
            self._buffer.clear()

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

            BatchingPlugin(self.batching_strategy).fill_buffer(self._buffer, self._batch_info, self._next_samples)

    def _generate_batch(self) -> list[BatchInput] | None:
        if self.batching_strategy == BatchingStrategy.NORMAL:
            return default_collate_fn(self._buffer, self._batch_info, self.renderer)
        else:
            # Non-NORMAL strategies (dynamic / padding_free) collate ragged pixel tensors with a
            # bare default_collate and have no vision-tower alignment, so multimodal data would
            # crash or hang. Fail loud instead of silently mishandling it.
            if any(k in s for s in self._buffer.samples for k in _MULTIMODAL_PASSTHROUGH_KEYS):
                raise NotImplementedError(
                    f"batching_strategy={self.batching_strategy.value!r} does not support multimodal data; "
                    "use the NORMAL strategy for image/video training."
                )

            from ...plugins.trainer_plugins.batching import BatchingPlugin

            return BatchingPlugin(self.batching_strategy).generate_batch(self._buffer, self._batch_info)

    def _next_samples(self, restart: bool) -> list[ModelInput] | None:
        try:
            return next(self._data_iter)
        except StopIteration:
            if not restart:
                return None

            # Dynamic batching may restart the provider to fill one token-budgeted batch.
            self._data_iter = iter(self._data_provider)
            try:
                return next(self._data_iter)
            except StopIteration:
                return None

    def state_dict(self) -> dict[str, Any]:
        return {
            "buffer": self._buffer.state_dict(),
            "data_provider": self._data_provider.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._buffer.load_state_dict(state["buffer"])
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
