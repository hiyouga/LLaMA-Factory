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

from llamafactory.data.collator import prepare_4d_attention_mask


def test_4d_attention_mask():
    o = 0.0
    x = torch.finfo(torch.float16).min
    attention_mask_with_indices = torch.tensor(
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
    )
    attention_mask_computed = prepare_4d_attention_mask(attention_mask_with_indices, torch.float16)
    attention_mask_expected = torch.tensor(
        [
            [
                [
                    [o, x, x, x, x, x],
                    [o, o, x, x, x, x],
                    [x, x, o, x, x, x],
                    [x, x, o, o, x, x],
                    [x, x, o, o, o, x],
                    [x, x, x, x, x, x],
                ]
            ],
            [
                [
                    [o, x, x, x, x, x],
                    [x, o, x, x, x, x],
                    [x, o, o, x, x, x],
                    [x, x, x, o, x, x],
                    [x, x, x, o, o, x],
                    [x, x, x, o, o, o],
                ]
            ],
        ],
        dtype=torch.float16,
    )
    assert list(attention_mask_computed.size()) == [2, 1, 6, 6]
    assert torch.all(attention_mask_computed == attention_mask_expected)
