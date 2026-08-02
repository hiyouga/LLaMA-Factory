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

from llamafactory.hparams import GeneratingArguments


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_top_p(top_p: float):
    with pytest.raises(ValueError, match=r"`top_p` must be in the range \(0\.0, 1\.0\]\."):
        GeneratingArguments(top_p=top_p)


@pytest.mark.parametrize("top_p", [1e-6, 0.7, 1.0])
def test_valid_top_p(top_p: float):
    generating_args = GeneratingArguments(top_p=top_p)
    assert generating_args.top_p == top_p
