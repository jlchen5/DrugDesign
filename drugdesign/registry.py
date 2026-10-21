"""A tiny name -> factory registry used to make featurizers and models pluggable."""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a string key to a factory callable.

    Usage::

        FEATURIZERS = Registry("featurizer")

        @FEATURIZERS.register("my_fp")
        class MyFingerprint: ...

        FEATURIZERS.create("my_fp", radius=3)
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, Callable[..., T]] = {}

    def register(
        self, name: Optional[str] = None, *, overwrite: bool = False
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if not callable(factory):
                raise ValueError("Registry entries must be callable factories")
            key = factory.__name__ if name is None else name
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Registry names must be non-empty strings")
            key = key.strip().lower()
            if key in self._items and not overwrite:
                raise KeyError(f"{self.kind} '{key}' is already registered")
            self._items[key] = factory
            return factory

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Registry names must be non-empty strings")
        key = name.strip().lower()
        if key not in self._items:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {available}")
        return self._items[key]

    def create(self, name: str, **params) -> T:
        return self.get(name)(**params)

    def names(self) -> list:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip().lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())
