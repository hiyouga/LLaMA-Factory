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

from llamafactory.v1.config import DataArguments, ModelArguments, TrainingArguments
from llamafactory.v1.core.data_engine import DataEngine
from llamafactory.v1.core.model_engine import ModelEngine
from llamafactory.v1.core.utils.batching import BatchGenerator
from llamafactory.v1.plugins.trainer_plugins.batching import (
    BatchingPlugin,
    _get_dynamic_micro_batch_sizes,
    _get_dynamic_padding_free_micro_batch_sizes,
)
from llamafactory.v1.utils.constants import IGNORE_INDEX
from llamafactory.v1.utils.objects import StatefulBuffer


def _make_model_input(length: int, start: int = 0):
    input_ids = list(range(start, start + length))
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * length,
        "labels": input_ids.copy(),
        "loss_weights": [1.0] * length,
    }


class _RestartableDataProvider:
    def __init__(self, batches):
        self.batches = batches
        self.num_iters = 0

    def __iter__(self):
        self.num_iters += 1
        return iter(self.batches)


def test_padding_free():
    buffer = StatefulBuffer()
    # Input samples:
    #   sample 0 input_ids: [0, 1]
    #   sample 1 input_ids: [10, 11, 12, 13]
    buffer.put([_make_model_input(2, 0), _make_model_input(4, 10)])
    batch_info = {"micro_batch_size": 2, "num_micro_batch": 1, "cutoff_len": 3}

    batch = BatchingPlugin("padding_free").generate_batch(buffer, batch_info)

    # Output batch:
    #   sample 1 is truncated to [10, 11, 12]
    #   both samples are packed into one sequence: [[0, 1, 10, 11, 12]]
    assert batch is not None
    assert len(batch) == 1
    assert batch[0]["input_ids"].shape == (1, 5)
    assert batch[0]["input_ids"].tolist() == [[0, 1, 10, 11, 12]]
    assert batch[0]["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert batch[0]["position_ids"].tolist() == [[0, 1, 0, 1, 2]]
    assert batch[0]["labels"].tolist() == [[0, 1, IGNORE_INDEX, 11, 12]]
    assert batch[0]["loss_weights"].tolist() == [[1.0, 1.0, 0.0, 1.0, 1.0]]
    assert len(buffer) == 0


def test_batching_plugin_data_provider_batch_sizes():
    batch_info = {
        "micro_batch_size": 2,
        "num_micro_batch": 3,
        "cutoff_len": 10,
    }

    assert BatchingPlugin("padding_free").get_data_provider_batch_size(batch_info) == 6
    assert BatchingPlugin("dynamic_batching").get_data_provider_batch_size(batch_info) == 1
    assert BatchingPlugin("dynamic_padding_free").get_data_provider_batch_size(batch_info) == 1


def test_dynamic_batching():
    # Input samples:
    #   sample lengths: [3, 4, 6, 2, 8, 9]
    #   input_ids:
    #     [0, 1, 2]
    #     [10, 11, 12, 13]
    #     [20, 21, 22, 23, 24, 25]
    #     [30, 31]
    #     [40, 41, 42, 43, 44, 45, 46, 47]
    #     [50, 51, 52, 53, 54, 55, 56, 57, 58]
    samples = [
        _make_model_input(3, 0),
        _make_model_input(4, 10),
        _make_model_input(6, 20),
        _make_model_input(2, 30),
        _make_model_input(8, 40),
        _make_model_input(9, 50),
    ]
    batch_info = {"micro_batch_size": 2, "num_micro_batch": 1, "cutoff_len": 10}

    # Dynamic batching output plan:
    #   dynamic batching reads one sample at a time and uses cutoff_len * micro_batch_size
    #   as the padded-token budget for one training micro batch.
    #   [3, 4, 6] fits within budget 20 as shape [3, 6]; adding [2] would exceed it.
    assert _get_dynamic_micro_batch_sizes(samples, batch_info) == [3]

    buffer = StatefulBuffer()
    buffer.put(samples)
    batch = BatchingPlugin("dynamic_batching").generate_batch(buffer, batch_info)

    assert batch is not None
    assert len(batch) == 1
    assert batch[0]["input_ids"].shape == (3, 6)
    assert batch[0]["input_ids"].tolist()[0] == [0, 1, 2, 0, 0, 0]
    assert len(buffer) == 3


def test_dynamic_batching_returns_none_when_token_budget_is_incomplete():
    buffer = StatefulBuffer()
    # Input buffer:
    #   only one sample with length [6].
    #   cutoff_len * micro_batch_size gives a padded-token budget of 20.
    #   this buffer has not filled the budget and has no next sample to prove overflow,
    #   so dynamic batching cannot produce a batch yet.
    buffer.put([_make_model_input(6, 0)])
    batch_info = {"micro_batch_size": 2, "num_micro_batch": 1, "cutoff_len": 10}

    assert _get_dynamic_micro_batch_sizes(buffer.samples, batch_info) == []
    assert BatchingPlugin("dynamic_batching").generate_batch(buffer, batch_info) is None
    # Batch generation does not read from the data iterator. It only returns None and keeps
    # existing samples in the buffer; BatchGenerator._fill_buffer handles refilling.
    assert len(buffer) == 1


def test_dynamic_batching_fill_buffer_restarts_until_micro_batch_is_complete():
    # Input data provider:
    #   each iterator pass yields one sample with length [6].
    #   each yielded item is a list[ModelInput], matching BatchGenerator._next_samples.
    #   _fill_buffer keeps restarting the iterator until the next appended sample
    #   proves that the previous dynamic micro batch has reached its budget boundary.
    samples = [_make_model_input(6, 0)]
    data_provider = _RestartableDataProvider([[sample] for sample in samples])

    batch_generator = BatchGenerator.__new__(BatchGenerator)
    batch_generator.batching_strategy = "dynamic_batching"
    batch_generator.micro_batch_size = 2
    batch_generator.num_micro_batch = 1
    batch_generator._buffer = StatefulBuffer()
    batch_generator._data_provider = data_provider
    batch_generator._data_iter = iter(data_provider)
    batch_generator._batch_info = {
        "micro_batch_size": 2,
        "num_micro_batch": 1,
        "cutoff_len": 10,
    }

    batch_generator._fill_buffer()

    # Filled buffer after restart:
    #   existing buffer [6, 6, 6] is kept; the fourth [6] remains for the next batch
    #   because adding it to the first dynamic micro batch would exceed the budget.
    assert data_provider.num_iters == 4
    assert _get_dynamic_micro_batch_sizes(batch_generator._buffer.samples, batch_generator._batch_info) == [3]

    batch = batch_generator._generate_batch()

    # Output batch:
    #   dynamic batching returns [micro_batch_0]
    #   micro_batch_0 consumes [6, 6, 6] => 3 samples, padded to shape [3, 6].
    assert batch is not None
    assert len(batch) == 1
    assert batch[0]["input_ids"].shape == (3, 6)
    assert len(batch_generator._buffer) == 1


def test_normal_batching():
    data_args = DataArguments(train_dataset="llamafactory/v1-sft-demo")
    data_engine = DataEngine(data_args.train_dataset)
    model_args = ModelArguments(model="llamafactory/tiny-random-qwen3")
    model_engine = ModelEngine(model_args=model_args)
    training_args = TrainingArguments(
        micro_batch_size=4,
        global_batch_size=8,
        cutoff_len=10,
        batching_workers=0,
        batching_strategy="normal",
    )
    batch_generator = BatchGenerator(
        data_engine,
        model_engine.renderer,
        micro_batch_size=training_args.micro_batch_size,
        global_batch_size=training_args.global_batch_size,
        cutoff_len=training_args.cutoff_len,
        batching_workers=training_args.batching_workers,
        batching_strategy=training_args.batching_strategy,
    )
    assert len(batch_generator) == len(data_engine) // training_args.global_batch_size
    batch = next(iter(batch_generator))
    assert len(batch) == 2
    assert batch[0]["input_ids"].shape == (4, 10)


def test_dynamic_padding_free():
    """Test core logic of dynamic padding free strategy: pack samples by total token budget without padding."""
    # Construct test samples (lengths: 3, 4, 6, 2, 8, 9)
    # input_ids breakdown:
    #   sample 0: [0,1,2] (length=3)
    #   sample 1: [10,11,12,13] (length=4)
    #   sample 2: [20,21,22,23,24,25] (length=6)
    #   sample 3: [30,31] (length=2)
    #   sample 4: [40-47] (length=8)
    #   sample 5: [50-58] (length=9)
    samples = [
        _make_model_input(3, 0),
        _make_model_input(4, 10),
        _make_model_input(6, 20),
        _make_model_input(2, 30),
        _make_model_input(8, 40),
        _make_model_input(9, 50),
    ]
    # Batch config: micro_batch_size=2 → token budget = cutoff_len * micro_batch_size = 10*2=20
    batch_info = {"micro_batch_size": 2, "num_micro_batch": 1, "cutoff_len": 10}

    # Budget=20: 3+4+6+2=15 ≤20 (adding 8 would exceed) → first 4 samples are selected
    assert _get_dynamic_padding_free_micro_batch_sizes(samples, batch_info) == [4]

    buffer = StatefulBuffer()
    buffer.put(samples)
    batch = BatchingPlugin("dynamic_padding_free").generate_batch(buffer, batch_info)

    assert batch is not None
    assert len(batch) == 1  # num_micro_batch=1
    packed_batch = batch[0]

    # Total packed length: 3+4+6+2=15 → input_ids shape = (1,15) (no padding)
    assert packed_batch["input_ids"].shape == (1, 15)

    # Verify input_ids concatenation (first label of non-initial samples set to IGNORE_INDEX)
    assert packed_batch["input_ids"].tolist() == [
        [
            0,
            1,
            2,  # Sample 0
            10,
            11,
            12,
            13,  # Sample 1
            20,
            21,
            22,
            23,
            24,
            25,  # Sample 2
            30,
            31,
        ]  # Sample 3
    ]

    # Verify labels (first token of non-initial samples is IGNORE_INDEX)
    assert packed_batch["labels"].tolist() == [
        [
            0,
            1,
            2,  # Sample 0
            IGNORE_INDEX,
            11,
            12,
            13,  # Sample 1
            IGNORE_INDEX,
            21,
            22,
            23,
            24,
            25,  # Sample 2
            IGNORE_INDEX,
            31,
        ]  # Sample 3
    ]

    # Verify attention_mask
    assert packed_batch["attention_mask"].tolist() == [[1] * 15]

    # Verify position_ids
    assert packed_batch["position_ids"].tolist() == [
        [
            0,
            1,
            2,  # Sample 0
            0,
            1,
            2,
            3,  # Sample 1
            0,
            1,
            2,
            3,
            4,
            5,  # Sample 2
            0,
            1,
        ]  # Sample 3
    ]

    # Verify remaining samples in buffer: 6-4=2 samples (length 8,9)
    assert len(buffer) == 2


def test_dynamic_padding_free_returns_none_when_token_budget_is_incomplete():
    buffer = StatefulBuffer()
    buffer.put([_make_model_input(6, 0)])
    batch_info = {"micro_batch_size": 2, "num_micro_batch": 1, "cutoff_len": 10}

    assert _get_dynamic_micro_batch_sizes(buffer.samples, batch_info) == []
    assert BatchingPlugin("dynamic_padding_free").generate_batch(buffer, batch_info) is None
    # Batch generation does not read from the data iterator. It only returns None and keeps
    # existing samples in the buffer; BatchGenerator._fill_buffer handles refilling.
    assert len(buffer) == 1


def test_dynamic_padding_free_fill_buffer_restarts_until_micro_batch_is_complete():
    """Test fill_buffer logic for dynamic_padding_free: restart data iterator until token budget is full.

    Data provider yields one sample of length 6 per iteration.
    _fill_buffer keeps restarting iterator until next sample exceeds budget.
    Budget = 2 * 10 = 20 tokens.
    3 samples (6*3=18) fit; 4th sample (24) exceeds budget.
    So buffer will have 4 samples after fill_buffer.
    """
    samples = [_make_model_input(6, 0)]
    data_provider = _RestartableDataProvider([[sample] for sample in samples])

    batch_generator = BatchGenerator.__new__(BatchGenerator)
    batch_generator.batching_strategy = "dynamic_padding_free"
    batch_generator.micro_batch_size = 2
    batch_generator.num_micro_batch = 1
    batch_generator._buffer = StatefulBuffer()
    batch_generator._data_provider = data_provider
    batch_generator._data_iter = iter(data_provider)
    batch_generator._batch_info = {
        "micro_batch_size": 2,
        "num_micro_batch": 1,
        "cutoff_len": 10,
    }

    # Execute fill buffer (will restart iterator multiple times to collect enough samples)
    batch_generator._fill_buffer()

    # Buffer after restarts:
    #   3 samples can fit (18 tokens)
    #   4th sample is kept in buffer for next batch
    #   => num_iters = 4
    assert data_provider.num_iters == 4
    assert _get_dynamic_padding_free_micro_batch_sizes(
        batch_generator._buffer.samples, batch_generator._batch_info
    ) == [3]

    batch = batch_generator._generate_batch()

    # Output batch:
    #   dynamic_padding_free returns [micro_batch_0]
    #   3 samples packed into shape [1, 18]
    assert batch is not None
    assert len(batch) == 1
    assert batch[0]["input_ids"].shape == (1, 18)
    assert len(batch_generator._buffer) == 1
