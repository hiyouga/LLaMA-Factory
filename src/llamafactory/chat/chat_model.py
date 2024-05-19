import asyncio
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence

from ..extras.misc import torch_gc
from ..hparams import get_infer_args
from .hf_engine import HuggingfaceEngine
from .vllm_engine import VllmEngine


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .base_engine import BaseEngine, Response


def _start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ChatModel:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        if model_args.infer_backend == "huggingface":
            self.engine: "BaseEngine" = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        elif model_args.infer_backend == "vllm":
            self.engine: "BaseEngine" = VllmEngine(model_args, data_args, finetuning_args, generating_args)
        else:
            raise NotImplementedError("Unknown backend: {}".format(model_args.infer_backend))

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self.engine.start(), self._loop)

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> List["Response"]:
        task = asyncio.run_coroutine_threadsafe(self.achat(messages, system, tools, image, **input_kwargs), self._loop)
        return task.result()

    async def achat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> List["Response"]:
        return await self.engine.chat(messages, system, tools, image, **input_kwargs)

    def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        generator = self.astream_chat(messages, system, tools, image, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        async for new_token in self.engine.stream_chat(messages, system, tools, image, **input_kwargs):
            yield new_token

    def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        task = asyncio.run_coroutine_threadsafe(self.aget_scores(batch_input, **input_kwargs), self._loop)
        return task.result()

    async def aget_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        return await self.engine.get_scores(batch_input, **input_kwargs)


def run_chat() -> None:
    try:
        import platform

        if platform.system() != "Windows":
            import readline  # noqa: F401
    except ImportError:
        print("Install `readline` for a better experience.")

    chat_model = ChatModel()
    messages = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            torch_gc()
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})
