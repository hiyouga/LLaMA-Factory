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

import pytest
from pydantic import ValidationError

from llamafactory.api.protocol import ChatCompletionRequest


MESSAGES = [{"role": "user", "content": "Hello"}]


@pytest.mark.parametrize("n", [0, -1])
def test_chat_completion_request_rejects_non_positive_response_count(n):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="test", messages=MESSAGES, n=n)


def test_chat_completion_request_allows_multiple_non_streamed_responses():
    request = ChatCompletionRequest(model="test", messages=MESSAGES, n=2)

    assert request.n == 2
