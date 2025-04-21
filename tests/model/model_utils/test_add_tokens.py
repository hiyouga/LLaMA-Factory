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

import pytest

from llamafactory.hparams import ModelArguments
from llamafactory.model import load_tokenizer


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

UNUSED_TOKEN = "<|UNUSED_TOKEN|>"


@pytest.mark.parametrize("special_tokens", [False, True])
def test_add_tokens(special_tokens: bool):
    if special_tokens:
        model_args = ModelArguments(model_name_or_path=TINY_LLAMA3, add_special_tokens=UNUSED_TOKEN)
    else:
        model_args = ModelArguments(model_name_or_path=TINY_LLAMA3, add_tokens=UNUSED_TOKEN)

    tokenizer = load_tokenizer(model_args)["tokenizer"]
    encoded_ids = tokenizer.encode(UNUSED_TOKEN, add_special_tokens=False)
    assert len(encoded_ids) == 1
    decoded_str = tokenizer.decode(encoded_ids, skip_special_tokens=True)
    if special_tokens:
        assert decoded_str == ""
    else:
        assert decoded_str == UNUSED_TOKEN


if __name__ == "__main__":
    pytest.main([__file__])
