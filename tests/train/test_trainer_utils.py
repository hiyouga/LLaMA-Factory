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

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.trainer_utils import asft_loss_func


def test_asft_loss_with_fully_masked_labels():
    policy_logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]]],
        requires_grad=True,
    )
    ref_logits = torch.ones_like(policy_logits)
    labels = torch.full((1, 3), IGNORE_INDEX)

    loss = asft_loss_func({"logits": policy_logits}, labels, ref_logits)

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert torch.isfinite(policy_logits.grad).all()
    assert torch.count_nonzero(policy_logits.grad) == 0
