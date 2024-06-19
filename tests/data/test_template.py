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

import os

from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")


def test_jinja_template():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    get_template_and_fix_tokenizer(tokenizer, name="llama3")
    assert tokenizer.chat_template != ref_tokenizer.chat_template

    messages = [
        {"role": "user", "content": "hi!"},
        {"role": "assistant", "content": "hello there"},
    ]
    assert tokenizer.apply_chat_template(messages) == ref_tokenizer.apply_chat_template(messages)
