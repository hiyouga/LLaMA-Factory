import asyncio
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence

from ..hparams import get_infer_args
from .hf_engine import HuggingfaceEngine
from .vllm_engine import VllmEngine


if TYPE_CHECKING:
    from .base_engine import BaseEngine, Response


class ChatModel:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        if model_args.infer_backend == "hf":
            self.engine: "BaseEngine" = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        elif model_args.infer_backend == "vllm":
            self.engine: "BaseEngine" = VllmEngine(model_args, data_args, finetuning_args, generating_args)
        else:
            raise NotImplementedError("Unknown backend: {}".format(model_args.infer_backend))

    def _get_event_loop():
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.new_event_loop()

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs,
    ) -> List["Response"]:
        loop = self._get_event_loop()
        return loop.run_until_complete(self.achat(messages, system, tools, **input_kwargs))

    async def achat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs,
    ) -> List["Response"]:
        return await self.engine.chat(messages, system, tools, **input_kwargs)

    def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        loop = self._get_event_loop()
        generator = self.astream_chat(messages, system, tools, **input_kwargs)
        while True:
            try:
                yield loop.run_until_complete(generator.__anext__())
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        async for new_token in self.engine.stream_chat(messages, system, tools, **input_kwargs):
            yield new_token

    def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        loop = self._get_event_loop()
        return loop.run_until_complete(self.aget_scores(batch_input, **input_kwargs))

    async def aget_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        return await self.engine.get_scores(batch_input, **input_kwargs)
