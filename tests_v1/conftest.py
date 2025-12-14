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
from pytest import Config, Item

from llamafactory.v1.utils.packages import is_transformers_version_greater_than


def pytest_collection_modifyitems(config: Config, items: list[Item]):
    if is_transformers_version_greater_than("4.57.0"):
        return

    skip_bc = pytest.mark.skip(reason="Skip backward compatibility tests")

    for item in items:
        if "tests_v1" in str(item.fspath):
            item.add_marker(skip_bc)
