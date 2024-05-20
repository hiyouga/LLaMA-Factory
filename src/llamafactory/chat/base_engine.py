from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Literal, Optional, Sequence, Union


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from vllm import AsyncLLMEngine

    from ..data import Template
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


@dataclass
class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]


class BaseEngine(ABC):
    model: Union["PreTrainedModel", "AsyncLLMEngine"]
    tokenizer: "PreTrainedTokenizer"
    can_generate: bool
    template: "Template"
    generating_args: Dict[str, Any]

    @abstractmethod
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None: ...

    @abstractmethod
    async def start(
        self,
    ) -> None: ...

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> List["Response"]: ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]: ...
