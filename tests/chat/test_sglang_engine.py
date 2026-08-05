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

import asyncio
from types import SimpleNamespace

from llamafactory.chat import sglang_engine


class _FakeResponse:
    status_code = 200

    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.closed = True

    def iter_lines(self, decode_unicode=False):
        yield b'data: {"text": "partial"}'
        yield b"data: [DONE]"


def test_generate_closes_response_when_stream_stops_early(monkeypatch):
    response = _FakeResponse()
    monkeypatch.setattr(sglang_engine.requests, "post", lambda *args, **kwargs: response)

    engine = object.__new__(sglang_engine.SGLangEngine)
    engine.base_url = "http://localhost:30000"
    engine.generating_args = {
        "max_new_tokens": 8,
        "repetition_penalty": 1.0,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 50,
        "skip_special_tokens": True,
    }
    engine.lora_request = False
    engine.processor = None
    engine.tokenizer = SimpleNamespace()
    engine.template = SimpleNamespace(
        mm_plugin=SimpleNamespace(process_messages=lambda messages, *args: messages),
        encode_oneturn=lambda *args: ([1, 2], None),
        get_stop_token_ids=lambda tokenizer: [3],
    )

    generator = asyncio.run(engine._generate([{"role": "user", "content": "hello"}]))
    assert next(generator) == {"text": "partial"}
    generator.close()

    assert response.closed
