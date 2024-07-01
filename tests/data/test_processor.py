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

from typing import Tuple

import pytest

from llamafactory.data.processors.processor_utils import infer_seqlen


@pytest.mark.parametrize(
    "test_input,test_output",
    [
        ((3000, 2000, 1000), (600, 400)),
        ((2000, 3000, 1000), (400, 600)),
        ((1000, 100, 1000), (900, 100)),
        ((100, 1000, 1000), (100, 900)),
    ],
)
def test_infer_seqlen(test_input: Tuple[int, int, int], test_output: Tuple[int, int]):
    assert test_output == infer_seqlen(*test_input)
