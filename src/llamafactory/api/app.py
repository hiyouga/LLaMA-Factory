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
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated, Optional

from ..chat import ChatModel
from ..extras.constants import EngineName
from ..extras.misc import torch_gc
from ..extras.packages import is_fastapi_available, is_starlette_available, is_uvicorn_available
from .chat import (
    create_chat_completion_response,
    create_score_evaluation_response,
    create_stream_chat_completion_response,
)
from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList,
    ScoreEvaluationRequest,
    ScoreEvaluationResponse,
)


if is_fastapi_available():
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer


if is_starlette_available():
    from sse_starlette import EventSourceResponse


if is_uvicorn_available():
    import uvicorn


async def sweeper() -> None:
    while True:
        torch_gc()
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: "FastAPI", chat_model: "ChatModel"):  # collects GPU memory
    if chat_model.engine.name == EngineName.HF:
        asyncio.create_task(sweeper())

    yield
    torch_gc()


def create_app(chat_model: "ChatModel") -> "FastAPI":
    root_path = os.getenv("FASTAPI_ROOT_PATH", "")
    app = FastAPI(lifespan=partial(lifespan, chat_model=chat_model), root_path=root_path)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_key = os.getenv("API_KEY")
    security = HTTPBearer(auto_error=False)

    async def verify_api_key(auth: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]):
        if api_key and (auth is None or auth.credentials != api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    @app.get(
        "/v1/models",
        response_model=ModelList,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def list_models():
        model_card = ModelCard(id=os.getenv("API_MODEL_NAME", "gpt-3.5-turbo"))
        return ModelList(data=[model_card])

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def create_chat_completion(request: ChatCompletionRequest):
        if not chat_model.engine.can_generate:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        if request.stream:
            generate = create_stream_chat_completion_response(request, chat_model)
            return EventSourceResponse(generate, media_type="text/event-stream", sep="\n")
        else:
            return await create_chat_completion_response(request, chat_model)

    @app.post(
        "/v1/score/evaluation",
        response_model=ScoreEvaluationResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def create_score_evaluation(request: ScoreEvaluationRequest):
        if chat_model.engine.can_generate:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        return await create_score_evaluation_response(request, chat_model)

    return app


def run_api() -> None:
    chat_model = ChatModel()
    app = create_app(chat_model)
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    print(f"Visit http://localhost:{api_port}/docs for API document.")
    uvicorn.run(app, host=api_host, port=api_port)
