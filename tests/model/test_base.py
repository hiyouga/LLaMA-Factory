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

import pytest

from llamafactory.train.test_utils import (
    compare_model,
    load_infer_model,
    load_reference_model,
    patch_valuehead_model,
)


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

TINY_LLAMA_VALUEHEAD = os.environ.get("TINY_LLAMA_VALUEHEAD", "llamafactory/tiny-random-Llama-3-valuehead")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "template": "llama3",
    "infer_dtype": "float16",
}


@pytest.fixture
def fix_valuehead_cpu_loading():
    patch_valuehead_model()


def test_base():
    model = load_infer_model(**INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA)
    compare_model(model, ref_model)


@pytest.mark.usefixtures("fix_valuehead_cpu_loading")
def test_valuehead():
    model = load_infer_model(add_valuehead=True, **INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA_VALUEHEAD, add_valuehead=True)
    compare_model(model, ref_model)
