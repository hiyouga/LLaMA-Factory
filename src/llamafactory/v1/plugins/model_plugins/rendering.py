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

import importlib

from ...utils import logging
from ...utils.plugin import BasePlugin
from ...utils.types import Message, ModelInput, Processor


logger = logging.get_logger(__name__)


class RenderingPlugin(BasePlugin):
    _attempted_template_imports: set[str] = set()

    def _ensure_template_imported(self) -> None:
        if self.name is None or self.name in self._attempted_template_imports:
            return

        full_module_name = f"{__package__}.templates.{self.name}"
        self._attempted_template_imports.add(self.name)
        try:
            importlib.import_module(full_module_name)
        except Exception as exc:
            logger.warning(f"[Template Registry] Failed to import {full_module_name}: {exc}")

    def __getitem__(self, method_name: str):
        self._ensure_template_imported()
        return super().__getitem__(method_name)

    def render_messages(
        self,
        processor: Processor,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        enable_thinking: bool = False,
    ) -> ModelInput:
        """Render messages in the template format."""
        return self["render_messages"](processor, messages, tools, is_generate, enable_thinking)

    def parse_messages(self, generated_text: str) -> Message:
        """Parse messages in the template format."""
        return self["parse_messages"](generated_text)
