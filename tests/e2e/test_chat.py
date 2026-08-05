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

import asyncio
import os
from threading import Thread

import pytest

from llamafactory.chat import ChatModel
from llamafactory.chat.chat_model import _start_background_loop


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
    "do_sample": False,
    "max_new_tokens": 1,
}

MESSAGES = [
    {"role": "user", "content": "Hi"},
]

EXPECTED_RESPONSE = "_rho"


class _AsyncStream:
    def __init__(self):
        self.closed = False
        self.started = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.started:
            self.started = True
            return "first"

        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


@pytest.mark.runs_on(["cpu", "mps"])
def test_chat():
    chat_model = ChatModel(INFER_ARGS)
    assert chat_model.chat(MESSAGES)[0].response_text == EXPECTED_RESPONSE


@pytest.mark.runs_on(["cpu", "mps"])
def test_stream_chat():
    chat_model = ChatModel(INFER_ARGS)
    response = ""
    for token in chat_model.stream_chat(MESSAGES):
        response += token

    assert response == EXPECTED_RESPONSE


def test_stream_chat_closes_async_generator():
    stream = _AsyncStream()
    chat_model = object.__new__(ChatModel)
    chat_model.astream_chat = lambda *args, **kwargs: stream
    chat_model._loop = asyncio.new_event_loop()
    chat_model._thread = Thread(target=_start_background_loop, args=(chat_model._loop,), daemon=True)
    chat_model._thread.start()

    try:
        generator = chat_model.stream_chat([])
        assert next(generator) == "first"
        generator.close()
        assert stream.closed
    finally:
        chat_model._loop.call_soon_threadsafe(chat_model._loop.stop)
        chat_model._thread.join()
        chat_model._loop.close()
