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

"""Interface for rendering plugins."""

from abc import ABC, abstractmethod

from ....utils.plugin import BasePlugin, ensure_methods_implemented
from ....utils.types import Message, ModelInput, Processor


class BaseRendering(ABC):
    """Contract implemented by rendering plugin staticmethod groups."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        ensure_methods_implemented(cls)

    @staticmethod
    @abstractmethod
    def render_messages(
        processor: Processor,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        enable_thinking: bool = False,
    ) -> ModelInput:
        """Render messages in the template format."""

    @staticmethod
    @abstractmethod
    def parse_message(generated_text: str) -> Message:
        """Parse messages in the template format."""


class RenderingPlugin(BasePlugin):
    """Plugin family for rendering staticmethod groups (dispatched via BasePlugin.__getattr__)."""


@RenderingPlugin("qwen3").register()
class Qwen3Rendering(BaseRendering):
    @staticmethod
    def render_messages(
        processor: Processor,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        enable_thinking: bool = False,
    ) -> ModelInput:
        from .qwen3 import render_qwen3_messages

        return render_qwen3_messages(processor, messages, tools, is_generate, enable_thinking)

    @staticmethod
    def parse_message(generated_text: str) -> Message:
        from .qwen3 import parse_qwen3_message

        return parse_qwen3_message(generated_text)


@RenderingPlugin("qwen3_nothink").register()
class Qwen3NoThinkRendering(BaseRendering):
    @staticmethod
    def render_messages(
        processor: Processor,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        enable_thinking: bool = False,
    ) -> ModelInput:
        from .qwen3_nothink import render_qwen3_nothink_messages

        return render_qwen3_nothink_messages(processor, messages, tools, is_generate, enable_thinking)

    @staticmethod
    def parse_message(generated_text: str) -> Message:
        from .qwen3_nothink import parse_qwen3_nothink_message

        return parse_qwen3_nothink_message(generated_text)
