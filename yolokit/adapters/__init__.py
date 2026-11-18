"""Adapter registry keyed by output-tensor layout name."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yolokit.adapters.base import Adapter

REGISTRY: dict[str, type[Adapter]] = {}


def register(cls: type[Adapter]) -> type[Adapter]:
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate adapter name {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> type[Adapter]:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown layout {name!r}; available: {sorted(REGISTRY)}") from None


# populate the registry
from yolokit.adapters import dfl, e2e, objectness  # noqa: E402,F401
