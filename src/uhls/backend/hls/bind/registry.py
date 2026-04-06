"""Built-in registry for operation binders."""

from __future__ import annotations

from collections.abc import Callable

from .builtin import CompatibilityBinder, LeftEdgeBinder
from .interfaces import OperationBinder

_BUILTIN_BINDER_FACTORIES: dict[str, Callable[..., OperationBinder]] = {
    "compat": CompatibilityBinder,
    "left_edge": LeftEdgeBinder,
}


def builtin_binder_names() -> tuple[str, ...]:
    """Return built-in operation binder names."""
    return tuple(sorted(_BUILTIN_BINDER_FACTORIES))


def create_builtin_binder(name: str, **binder_kwargs: object) -> OperationBinder:
    """Instantiate one built-in operation binder."""
    normalized = name.strip().lower().replace("-", "_")
    try:
        return _BUILTIN_BINDER_FACTORIES[normalized](**binder_kwargs)
    except KeyError as exc:
        supported = ", ".join(builtin_binder_names())
        raise ValueError(f"unsupported operation binder '{name}'; expected one of: {supported}") from exc
