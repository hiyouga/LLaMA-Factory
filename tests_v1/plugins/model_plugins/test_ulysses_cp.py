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

import pytest
import torch
import torch.multiprocessing as mp

from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.core.model_engine import ModelEngine
from llamafactory.v1.plugins.model_plugins.parallelization.sequence_parallel import (
    SequenceParallelModelPlugin,
    sequence_parallel_loss,
)
from llamafactory.v1.utils.env import find_available_port
from llamafactory.v1.utils.pytest import dist_env


def _test_sequence_parallel_loss(local_rank: int, world_size: int, master_port: int, cp_size: int, dp_size: int):
    with dist_env(local_rank, world_size, master_port):
        model_args = ModelArguments(model="llamafactory/tiny-random-qwen3")

        # Initialize distributed interface with config
        dist_config = {"cp_mode": "ulysses", "cp_size": cp_size, "dp_size": dp_size}
        DistributedInterface(dist_config)

        # Now create model engine
        model_engine = ModelEngine(model_args=model_args)

        # Apply sequence parallel plugin
        SequenceParallelModelPlugin(dist_config.get("cp_mode", "ulysses"))(model_engine.model, dist_config)

        model_inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "position_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "loss_weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]),
        }

        loss = sequence_parallel_loss(model_engine.model, model_inputs)
        assert loss is not None


@pytest.mark.runs_on(["cuda", "npu"])
@pytest.mark.require_distributed(2)
@pytest.mark.parametrize("cp_size, dp_size", [(2, 1)])
def test_sequence_parallel_loss(cp_size, dp_size):
    master_port = find_available_port()
    world_size = cp_size * dp_size
    mp.spawn(_test_sequence_parallel_loss, args=(world_size, master_port, cp_size, dp_size), nprocs=world_size)
