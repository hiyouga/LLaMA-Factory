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
from unittest.mock import Mock

import llamafactory.api.app as api_app
from llamafactory.extras.constants import EngineName


def test_lifespan_cancels_sweeper(monkeypatch):
    created_tasks = []
    create_task = asyncio.create_task

    def track_task(coroutine):
        task = create_task(coroutine)
        created_tasks.append(task)
        return task

    async def exercise_lifespan():
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def fake_sweeper():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        monkeypatch.setattr(api_app, "sweeper", fake_sweeper)
        monkeypatch.setattr(api_app.asyncio, "create_task", track_task)
        torch_gc = Mock()
        monkeypatch.setattr(api_app, "torch_gc", torch_gc)
        chat_model = SimpleNamespace(engine=SimpleNamespace(name=EngineName.HF))

        async with api_app.lifespan(None, chat_model):
            await started.wait()

        assert len(created_tasks) == 1
        assert created_tasks[0].done()
        assert created_tasks[0].cancelled()
        assert stopped.is_set()
        torch_gc.assert_called_once_with()

    asyncio.run(exercise_lifespan())
