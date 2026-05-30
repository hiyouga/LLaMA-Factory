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

"""Lightweight plugin registry with per-family isolation and typed params."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import fields
from typing import Any

from . import logging


logger = logging.get_logger(__name__)


def ensure_methods_implemented(cls: type) -> None:
    """Raise if *cls* leaves any required abstract method unimplemented.

    Meant to be called from a contract base's ``__init_subclass__`` so that
    incomplete *implementation* classes fail at definition time, even though
    staticmethod groups are never instantiated (which is when ABC normally
    enforces ``@abstractmethod``).

    Reads the required names from the MRO parents' ``__abstractmethods__``
    rather than ``cls.__abstractmethods__``: the latter is not populated yet
    while ``__init_subclass__`` runs (``ABCMeta`` computes it after
    ``type.__new__``). A contract base that defines ``__init_subclass__`` is
    never checked against itself, because ``__init_subclass__`` does not fire
    for the class that defines it.
    """
    required: set[str] = set()
    for base in cls.__mro__[1:]:
        required |= getattr(base, "__abstractmethods__", frozenset())

    missing = sorted(name for name in required if getattr(getattr(cls, name, None), "__isabstractmethod__", False))
    if missing:
        raise TypeError(f"{cls.__name__} does not implement all required methods: {missing}")


class BasePlugin:
    """Base class for plugin families.

    Routing shape is a flat ``_registry[plugin_name] = object``. Each family
    subclass gets its own registry bucket so one family cannot accidentally see
    another family's registrations. A registered entry is a single object:

    * a **function** for single-function families, called via ``Plugin(name)(...)``;
    * a **staticmethod group class** for method-group families, called via
      ``Plugin(name).method(...)``.
    """

    _registry: dict[str, Any]
    _params: dict[str, type | None]
    _aliases: dict[str, dict[str, str] | None]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry = {}
        cls._params = {}
        cls._aliases = {}

    _registry = {}
    _params = {}
    _aliases = {}

    def __init__(self, name: str | None = None) -> None:
        self.name = name

    def register(
        self,
        *,
        params: type | None = None,
        aliases: dict[str, str] | None = None,
    ) -> Callable:
        """Decorator to register a callable or staticmethod-group class under ``self.name``."""
        if self.name is None:
            raise ValueError("Plugin name should be specified.")

        cls = type(self)
        if self.name in cls._registry:
            logger.warning_rank0_once(f"Plugin {self.name!r} is already registered.")

        if params is not None:
            existing = cls._params.get(self.name)
            if existing is not None and existing is not params:
                logger.warning_rank0_once(
                    f"Params for {cls.__name__}({self.name!r}) re-declared ({existing.__name__} -> {params.__name__})."
                )
            cls._params[self.name] = params

        if aliases is not None:
            existing_aliases = cls._aliases.get(self.name)
            if existing_aliases is not None and existing_aliases != aliases:
                logger.warning_rank0_once(f"Aliases for {cls.__name__}({self.name!r}) re-declared.")
            cls._aliases[self.name] = aliases

        def decorator(obj: Any) -> Any:
            cls._registry[self.name] = obj
            return obj

        return decorator

    def _resolve(self) -> Any:
        cls = type(self)
        if self.name is None:
            raise ValueError(f"{cls.__name__} must be constructed with a name.")

        if self.name not in cls._registry:
            raise ValueError(f"Plugin {self.name!r} is not registered under {cls.__name__}. Available: {cls.names()}")

        return cls._registry[self.name]

    def __call__(self, *args, **kwargs) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        # Only called when normal attribute lookup fails, so real instance and
        # class attributes are never shadowed by plugin methods.
        return getattr(self._resolve(), attr)

    @classmethod
    def names(cls) -> list[str]:
        """List plugin names registered under this family."""
        return sorted(cls._registry.keys())

    @classmethod
    def parse_params(cls, name: str, config: Any) -> Any:
        """Validate and convert config to the ParamsClass registered for name."""
        params_cls = cls._params.get(name)
        if params_cls is None:
            return config

        if isinstance(config, params_cls):
            return config

        if config is None:
            config = {}
        elif isinstance(config, str):
            config = {"name": config}
        elif not isinstance(config, dict):
            config = dict(config)

        aliases = cls._aliases.get(name) or {}
        resolved: dict[str, Any] = {}
        for key, value in config.items():
            canonical = aliases.get(key, key)
            if canonical in resolved:
                raise ValueError(
                    f"Conflicting keys for {cls.__name__}({name!r}): "
                    f"{key!r} and canonical key {canonical!r} are both present."
                )
            resolved[canonical] = value

        all_fields = fields(params_cls)
        known = {field.name for field in all_fields}

        unknown = set(resolved) - known - {"name"}
        if unknown:
            raise ValueError(
                f"Unknown params for {cls.__name__}({name!r}): {sorted(unknown)}. Expected keys: {sorted(known)}"
            )

        missing = [
            field.name
            for field in all_fields
            if field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING  # type: ignore[misc]
            and field.name not in resolved
        ]
        if missing:
            raise ValueError(f"Missing required params for {cls.__name__}({name!r}): {missing}.")

        valid = {key: value for key, value in resolved.items() if key in known}
        return params_cls(**valid)
