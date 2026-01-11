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


if __name__ == "__main__":
    """
    python -m tests_v1.core.utils.test_batching
    """
    test_normal_batching()
