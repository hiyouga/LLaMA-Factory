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


from collections.abc import Callable

from . import logging


logger = logging.get_logger(__name__)


class BasePlugin:
    """Base class for plugins.

    A plugin is a callable object that can be registered and called by name.
    """

    _registry: dict[str, Callable] = {}

    def __init__(self, name: str | None = None):
        """Initialize the plugin with a name.

        Args:
            name (str): The name of the plugin.
        """
        self.name = name

    @property
    def register(self):
        """Decorator to register a function as a plugin.

        Example usage:
        ```python
        @PrintPlugin("hello").register()
        def print_hello():
            print("Hello world!")
        ```
        """
        if self.name is None:
            raise ValueError("Plugin name is not specified.")

        if self.name in self._registry:
            logger.warning_rank0_once(f"Plugin {self.name} is already registered.")

        def decorator(func: Callable) -> Callable:
            self._registry[self.name] = func
            return func

        return decorator

    def __call__(self, *args, **kwargs):
        """Call the registered function with the given arguments.

        Example usage:
        ```python
        PrintPlugin("hello")()
        ```
        """
        if self.name not in self._registry:
            raise ValueError(f"Plugin {self.name} is not registered.")

        return self._registry[self.name](*args, **kwargs)


if __name__ == "__main__":
    """
    python -m llamafactory.v1.utils.plugin
    """

    class PrintPlugin(BasePlugin):
        pass

    @PrintPlugin("hello").register
    def print_hello():
        print("Hello world!")

    PrintPlugin("hello")()
