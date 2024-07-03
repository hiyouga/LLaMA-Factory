# Copyright 2024 the LlamaFactory team.
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

from llamafactory.model.model_utils.packing import get_seqlens_in_batch, get_unpad_data


def test_get_seqlens_in_batch():
    attention_mask_with_indices = torch.tensor(
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
    )
    seqlens_in_batch = get_seqlens_in_batch(attention_mask_with_indices)
    assert list(seqlens_in_batch.size()) == [5]
    assert torch.all(seqlens_in_batch == torch.tensor([2, 3, 1, 2, 3]))


def test_get_unpad_data():
    attention_mask_with_indices = torch.tensor(
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
    )
    indices, cu_seqlens, max_seqlen_in_batch = get_unpad_data(attention_mask_with_indices)
    assert torch.all(indices == torch.tensor([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]))
    assert torch.all(cu_seqlens == torch.tensor([0, 2, 5, 6, 8, 11], dtype=torch.int32))
    assert max_seqlen_in_batch == 3
