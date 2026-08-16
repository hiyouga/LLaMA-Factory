# Copyright 2026 the LlamaFactory team.
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

import torch

from llamafactory.train.dpo.trainer import get_batch_logps_memory_efficient
from llamafactory.train.trainer_utils import get_batch_logps


def test_get_batch_logps_memory_efficient_matches_baseline():
    torch.manual_seed(42)
    labels = torch.randint(0, 17, (4, 130))
    labels[:, :3] = -100
    baseline_logits = torch.randn(4, 130, 17, requires_grad=True)
    chunked_logits = baseline_logits.detach().clone().requires_grad_(True)

    expected_logps, expected_lengths = get_batch_logps(baseline_logits, labels)
    actual_logps, actual_lengths = get_batch_logps_memory_efficient(chunked_logits, labels)

    torch.testing.assert_close(actual_logps, expected_logps)
    torch.testing.assert_close(actual_lengths, expected_lengths)

    expected_logps.sum().backward()
    actual_logps.sum().backward()
    torch.testing.assert_close(chunked_logits.grad, baseline_logits.grad)
