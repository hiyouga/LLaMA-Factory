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


from collections import defaultdict
from collections.abc import Callable
from typing import Any

from . import logging


logger = logging.get_logger(__name__)


class BasePlugin:
    """Base class for plugins.

    A plugin is a callable object that can be registered and called by name.

    Example usage:
    ```python
    class PrintPlugin(BasePlugin):
        def again(self):  # optional
            self["again"]()


    @PrintPlugin("hello").register()
    def print_hello():
        print("Hello world!")


    @PrintPlugin("hello").register("again")
    def print_hello_again():
        print("Hello world! Again.")


    PrintPlugin("hello")()
    PrintPlugin("hello").again()
    ```
    """

    _registry: dict[str, dict[str, Callable]] = defaultdict(dict)

    def __init__(self, name: str | None = None) -> None:
        """Initialize the plugin with a name."""
        self.name = name

    def register(self, method_name: str = "__call__") -> Callable:
        """Decorator to register a function as a plugin."""
        if self.name is None:
            raise ValueError("Plugin name should be specified.")

        if method_name in self._registry[self.name]:
            logger.warning_rank0_once(f"Method {method_name} of plugin {self.name} is already registered.")

        def decorator(func: Callable) -> Callable:
            self._registry[self.name][method_name] = func
            return func

        return decorator

    def __call__(self, *args, **kwargs) -> Any:
        """Call the registered function with the given arguments."""
        return self["__call__"](*args, **kwargs)

    def __getattr__(self, method_name: str) -> Callable:
        """Get the registered function with the given name."""
        return self[method_name]

    def __getitem__(self, method_name: str) -> Callable:
        """Get the registered function with the given name."""
        if method_name not in self._registry[self.name]:
            raise ValueError(f"Method {method_name} of plugin {self.name} is not registered.")

        return self._registry[self.name][method_name]


if __name__ == "__main__":
    """
    python -m llamafactory.v1.utils.plugin
    """

    class PrintPlugin(BasePlugin):
        def again(self):  # optional
            self["again"]()

    @PrintPlugin("hello").register()
    def print_hello():
        print("Hello world!")

    @PrintPlugin("hello").register("again")
    def print_hello_again():
        print("Hello world! Again.")

    PrintPlugin("hello")()
    PrintPlugin("hello").again()
