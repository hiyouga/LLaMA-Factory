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

from . import logging


logger = logging.get_logger(__name__)


class BasePlugin:
    """Base class for plugins.

    A plugin is a callable object that can be registered and called by name.
    """

    _registry: dict[str, dict[str, Callable]] = defaultdict(dict)

    def __init__(self, name: str | None = None):
        """Initialize the plugin with a name.

        Args:
            name (str): The name of the plugin.
        """
        self.name = name

    def register(self, method_name: str = "__call__"):
        """Decorator to register a function as a plugin.

        Example usage:
        ```python
        @PrintPlugin("hello").register()
        def print_hello():
            print("Hello world!")


        @PrintPlugin("hello").register("again")
        def print_hello_again():
            print("Hello world! Again.")
        ```
        """
        if self.name is None:
            raise ValueError("Plugin name should be specified.")

        if method_name in self._registry[self.name]:
            logger.warning_rank0_once(f"Method {method_name} of plugin {self.name} is already registered.")

        def decorator(func: Callable) -> Callable:
            self._registry[self.name][method_name] = func
            return func

        return decorator

    def __call__(self, *args, **kwargs):
        """Call the registered function with the given arguments.

        Example usage:
        ```python
        PrintPlugin("hello")()
        ```
        """
        if "__call__" not in self._registry[self.name]:
            raise ValueError(f"Method __call__ of plugin {self.name} is not registered.")

        return self._registry[self.name]["__call__"](*args, **kwargs)

    def __getattr__(self, method_name: str):
        """Get the registered function with the given name.

        Example usage:
        ```python
        PrintPlugin("hello").again()
        ```
        """
        if method_name not in self._registry[self.name]:
            raise ValueError(f"Method {method_name} of plugin {self.name} is not registered.")

        return self._registry[self.name][method_name]


if __name__ == "__main__":
    """
    python -m llamafactory.v1.utils.plugin
    """

    class PrintPlugin(BasePlugin):
        pass

    @PrintPlugin("hello").register()
    def print_hello():
        print("Hello world!")

    @PrintPlugin("hello").register("again")
    def print_hello_again():
        print("Hello world! Again.")

    PrintPlugin("hello")()
    PrintPlugin("hello").again()
