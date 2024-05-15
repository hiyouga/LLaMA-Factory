import os
from contextlib import asynccontextmanager
from typing import Optional

from typing_extensions import Annotated

from ..chat import ChatModel
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


@asynccontextmanager
async def lifespan(app: "FastAPI"):  # collects GPU memory
    yield
    torch_gc()


def create_app(chat_model: "ChatModel") -> "FastAPI":
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_key = os.environ.get("API_KEY")
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
        model_card = ModelCard(id="gpt-3.5-turbo")
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
            return EventSourceResponse(generate, media_type="text/event-stream")
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
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8000"))
    print("Visit http://localhost:{}/docs for API document.".format(api_port))
    uvicorn.run(app, host=api_host, port=api_port)
