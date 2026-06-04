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
from collections.abc import Callable

from ...utils import logging
from ...utils.plugin import BasePlugin
from ...utils.types import Message


logger = logging.get_logger(__name__)


class RenderingPlugin(BasePlugin):
    """Override hook for the built-in :class:`~llamafactory.v1.core.utils.rendering.Renderer`.

    The default rendering path (``render_messages`` / ``parse_message``) lives in the
    core ``Renderer`` and is used as-is when nothing is registered here. To customize a
    step for a given template, register a replacement in source code::

        @RenderingPlugin("my_template").register("render_messages")
        def render_my_template(processor, messages, tools=None, *, is_generate=False, enable_thinking=False):
            ...
            return ModelInput(...)

    and construct the renderer with that name (``Renderer(processor, name="my_template")``).
    Methods left unregistered for a name fall back to the built-in default, so a template
    may override only ``parse_message`` and still use the default ``render_messages``.
    """

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

    def get(self, method_name: str) -> Callable | None:
        """Return the registered override for ``method_name``, or ``None`` if there is none.

        Unlike ``__getitem__`` this never raises, so the caller can cleanly fall back to
        the built-in default when no custom implementation is registered.
        """
        self._ensure_template_imported()
        if self.name is None:
            return None
        return self._registry[self.name].get(method_name)

    def render_messages(self, *args, **kwargs):
        """Render messages using a template-specific renderer."""
        return self["render_messages"](*args, **kwargs)

    def parse_message(self, generated_text: str) -> Message:
        """Parse generated text using a model-specific parser."""
        return self["parse_message"](generated_text)

