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

from llamafactory.v1.plugins.model_plugins.parallelization.ring import _use_ring_attn_ascend

if _use_ring_attn_ascend():
    pytest.importorskip("ring_attn_ascend")
else:
    pytest.importorskip("ring_flash_attn")
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


def _test_sequence_parallel_loss_ring(
    local_rank: int,
    world_size: int,
    master_port: int,
    cp_size: int,
    dp_size: int,
    batch_size: int,
    cp_mode: str,
):
    with dist_env(local_rank, world_size, master_port):
        model_args = ModelArguments(model="llamafactory/tiny-random-qwen3")

        dist_config = {"cp_mode": cp_mode, "cp_size": cp_size, "dp_size": dp_size}
        DistributedInterface(dist_config)

        model_engine = ModelEngine(model_args=model_args)

        SequenceParallelModelPlugin(cp_mode)(model_engine.model, dist_config)

        # seq_len=6 is divisible by cp_size=2; local len=3 is odd (ok for ring, not zigzag)
        # seq_len=8 gives local len=4 (even, required for zigzag)
        seq_len = 8 if cp_mode == "ring_zigzag" else 6
        input_ids = torch.arange(1, batch_size * seq_len + 1, dtype=torch.long).view(batch_size, seq_len)
        model_inputs = {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(1, seq_len + 1, dtype=torch.long).repeat(batch_size, 1),
            "loss_weights": torch.ones(batch_size, seq_len),
        }

        loss = sequence_parallel_loss(model_engine.model, model_inputs)
        assert loss is not None


@pytest.mark.runs_on(["cuda", "npu"])
@pytest.mark.require_distributed(2)
@pytest.mark.parametrize(("cp_mode", "cp_size", "dp_size", "batch_size"), [
    ("ring", 2, 1, 1),
    ("ring", 2, 1, 2),
    ("ring_zigzag", 2, 1, 1),
])
def test_sequence_parallel_loss_ring(cp_mode, cp_size, dp_size, batch_size):
    master_port = find_available_port()
    world_size = cp_size * dp_size
    mp.spawn(
        _test_sequence_parallel_loss_ring,
        args=(world_size, master_port, cp_size, dp_size, batch_size, cp_mode),
        nprocs=world_size,
    )
