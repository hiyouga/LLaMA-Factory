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

import os

import torch

from llamafactory.train.tuner import run_exp


def main():
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        local_rank = os.environ.get("SLURM_LOCALID")

    if local_rank is not None:
        try:
            local_rank_int = int(local_rank)
        except ValueError:
            local_rank_int = 0
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count() or 1
            torch.cuda.set_device(local_rank_int % device_count)
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
