"""Small, explicit plugin registries.

Every extension point of ResearchRL (trainer adapters, update rules, verifiers, recurrent
states, compute controllers, ...) has a named :class:`Registry`. Research plugins register
themselves with a decorator and are later looked up by the string name stored in an
experiment configuration, so a paper's code never has to edit framework files.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised for unknown or duplicate registry entries."""


class Registry(Generic[T]):
    """A name -> object mapping with decorator-based registration."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str | None = None, *, overwrite: bool = False) -> Callable[[T], T]:
        """Decorator registering ``obj`` under ``name`` (defaults to ``obj.__name__``)."""

        def decorator(obj: T) -> T:
            key = name if name is not None else getattr(obj, "__name__", None)
            if not key:
                raise RegistryError(f"cannot infer a {self.kind} name for {obj!r}; pass name=...")
            self.add(key, obj, overwrite=overwrite)
            return obj

        return decorator

    def add(self, name: str, obj: T, *, overwrite: bool = False) -> None:
        if name in self._entries and not overwrite and self._entries[name] is not obj:
            raise RegistryError(f"{self.kind} {name!r} is already registered")
        self._entries[name] = obj

    def get(self, name: str) -> T:
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(sorted(self._entries)) or "<none>"
            raise RegistryError(f"unknown {self.kind} {name!r}; known: {known}") from None

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Look up ``name`` and call it (useful when entries are classes or factories)."""
        factory = self.get(name)
        if not callable(factory):
            raise RegistryError(f"{self.kind} {name!r} is not callable")
        return factory(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"


# Framework-wide registries. Plugins import these and register into them.
TRAINERS: Registry[Any] = Registry("trainer adapter")
UPDATE_RULES: Registry[Any] = Registry("update rule")
VERIFIERS: Registry[Any] = Registry("verifier")
RECURRENT_STATES: Registry[Any] = Registry("recurrent state")
COMPUTE_CONTROLLERS: Registry[Any] = Registry("compute controller")
PARAMETER_PRIORS: Registry[Any] = Registry("parameter prior")
ENVIRONMENTS: Registry[Any] = Registry("agent environment")
ADVANTAGE_SHAPERS: Registry[Any] = Registry("advantage shaper")
