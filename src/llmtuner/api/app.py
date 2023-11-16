import json
from typing import List, Tuple
from pydantic import BaseModel
from contextlib import asynccontextmanager

from llmtuner.api.protocol import (
    Role,
    Finish,
    ModelCard,
    ModelList,
    ChatMessage,
    DeltaMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponseUsage
)
from llmtuner.chat import ChatModel
from llmtuner.extras.misc import torch_gc
from llmtuner.extras.packages import (
    is_fastapi_availble, is_starlette_available, is_uvicorn_available
)


if is_fastapi_availble():
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware


if is_starlette_available():
    from sse_starlette import EventSourceResponse


if is_uvicorn_available():
    import uvicorn


@asynccontextmanager
async def lifespan(app: "FastAPI"): # collects GPU memory
    yield
    torch_gc()


def to_json(data: BaseModel) -> str:
    try: # pydantic v2
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except: # pydantic v1
        return data.json(exclude_unset=True, ensure_ascii=False)


def create_app(chat_model: "ChatModel") -> "FastAPI":
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        model_card = ModelCard(id="gpt-3.5-turbo")
        return ModelList(data=[model_card])

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, status_code=status.HTTP_200_OK)
    async def create_chat_completion(request: ChatCompletionRequest):
        if len(request.messages) == 0 or request.messages[-1].role != Role.USER:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request")

        query = request.messages[-1].content
        prev_messages = request.messages[:-1]
        if len(prev_messages) and prev_messages[0].role == Role.SYSTEM:
            system = prev_messages.pop(0).content
        else:
            system = None

        history = []
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == Role.USER and prev_messages[i+1].role == Role.ASSISTANT:
                    history.append([prev_messages[i].content, prev_messages[i+1].content])
                else:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supports u/a/u/a/u...")
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supports u/a/u/a/u...")

        if request.stream:
            generate = predict(query, history, system, request)
            return EventSourceResponse(generate, media_type="text/event-stream")

        responses = chat_model.chat(
            query, history, system,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
            num_return_sequences=request.n
        )

        prompt_length, response_length = 0, 0
        choices = []
        for i, response in enumerate(responses):
            choices.append(ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role=Role.ASSISTANT, content=response.response_text),
                finish_reason=Finish.STOP if response.finish_reason == "stop" else Finish.LENGTH
            ))
            prompt_length = response.prompt_length
            response_length += response.response_length

        usage = ChatCompletionResponseUsage(
            prompt_tokens=prompt_length,
            completion_tokens=response_length,
            total_tokens=prompt_length+response_length
        )

        return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)

    async def predict(query: str, history: List[Tuple[str, str]], system: str, request: ChatCompletionRequest):
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role=Role.ASSISTANT),
            finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
        yield to_json(chunk)

        for new_text in chat_model.stream_chat(
            query, history, system,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens
        ):
            if len(new_text) == 0:
                continue

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
            yield to_json(chunk)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason=Finish.STOP
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
        yield to_json(chunk)
        yield "[DONE]"

    return app


if __name__ == "__main__":
    chat_model = ChatModel()
    app = create_app(chat_model)
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
