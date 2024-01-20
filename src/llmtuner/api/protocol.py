import time
from enum import Enum, unique
from typing import List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Literal


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@unique
class Finish(str, Enum):
    STOP = "stop"
    LENGTH = "length"


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: Literal["owner"] = "owner"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Role
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Role] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    do_sample: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    max_tokens: Optional[int] = None
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Finish


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Finish] = None


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: Literal["chatcmpl-default"] = "chatcmpl-default"
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


class ChatCompletionStreamResponse(BaseModel):
    id: Literal["chatcmpl-default"] = "chatcmpl-default"
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


class ScoreEvaluationRequest(BaseModel):
    model: str
    messages: List[str]
    max_length: Optional[int] = None


class ScoreEvaluationResponse(BaseModel):
    id: Literal["scoreeval-default"] = "scoreeval-default"
    object: Literal["score.evaluation"] = "score.evaluation"
    model: str
    scores: List[float]
