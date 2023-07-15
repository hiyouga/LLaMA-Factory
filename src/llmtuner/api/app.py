import uvicorn
from threading import Thread
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import TextIteratorStreamer
from contextlib import asynccontextmanager
from sse_starlette import EventSourceResponse
from typing import Any, Dict

from llmtuner.tuner import get_infer_args, load_model_and_tokenizer
from llmtuner.extras.misc import get_logits_processor, torch_gc
from llmtuner.extras.template import Template
from llmtuner.api.protocol import (
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


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    torch_gc()


def create_app():
    model_args, data_args, finetuning_args, generating_args = get_infer_args()
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)

    prompt_template = Template(data_args.prompt_template)
    source_prefix = data_args.source_prefix if data_args.source_prefix else ""

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
        global model_args
        model_card = ModelCard(id="gpt-3.5-turbo")
        return ModelList(data=[model_card])

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        if request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="Invalid request")
        query = request.messages[-1].content

        prev_messages = request.messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == "system":
            prefix = prev_messages.pop(0).content
        else:
            prefix = source_prefix

        history = []
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                    history.append([prev_messages[i].content, prev_messages[i+1].content])

        inputs = tokenizer([prompt_template.get_prompt(query, history, prefix)], return_tensors="pt")
        inputs = inputs.to(model.device)

        gen_kwargs = generating_args.to_dict()
        gen_kwargs.update({
            "input_ids": inputs["input_ids"],
            "temperature": request.temperature if request.temperature else gen_kwargs["temperature"],
            "top_p": request.top_p if request.top_p else gen_kwargs["top_p"],
            "logits_processor": get_logits_processor()
        })

        if request.max_tokens:
            gen_kwargs.pop("max_length", None)
            gen_kwargs["max_new_tokens"] = request.max_tokens

        if request.stream:
            generate = predict(gen_kwargs, request.model)
            return EventSourceResponse(generate, media_type="text/event-stream")

        generation_output = model.generate(**gen_kwargs)
        outputs = generation_output.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)

        usage = ChatCompletionResponseUsage(
            prompt_tokens=len(inputs["input_ids"][0]),
            completion_tokens=len(outputs),
            total_tokens=len(inputs["input_ids"][0]) + len(outputs)
        )

        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop"
        )

        return ChatCompletionResponse(model=request.model, choices=[choice_data], usage=usage, object="chat.completion")

    async def predict(gen_kwargs: Dict[str, Any], model_id: str):
        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield chunk.json(exclude_unset=True, ensure_ascii=False)

        for new_text in streamer:
            if len(new_text) == 0:
                continue

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
            yield chunk.json(exclude_unset=True, ensure_ascii=False)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason="stop"
        )
        chunk = ChatCompletionStreamResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield chunk.json(exclude_unset=True, ensure_ascii=False)
        yield "[DONE]"

    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
