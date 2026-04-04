"""Built-in scheduler registry for flat SGU algorithms."""

from __future__ import annotations

from collections.abc import Callable

from .builtin import ALAPScheduler, ASAPScheduler
from .interfaces import SGUScheduler

_BUILTIN_SCHEDULER_FACTORIES: dict[str, Callable[..., SGUScheduler]] = {
    "alap": ALAPScheduler,
    "asap": ASAPScheduler,
}


def builtin_scheduler_names() -> tuple[str, ...]:
    """Return built-in flat SGU scheduler names."""
    return tuple(sorted(_BUILTIN_SCHEDULER_FACTORIES))


def create_builtin_scheduler(name: str, **scheduler_kwargs: object) -> SGUScheduler:
    """Instantiate one built-in flat SGU scheduler."""
    normalized = name.strip().lower().replace("-", "_")
    try:
        return _BUILTIN_SCHEDULER_FACTORIES[normalized](**scheduler_kwargs)
    except KeyError as exc:
        supported = ", ".join(builtin_scheduler_names())
        raise ValueError(f"unsupported flat SGU scheduler '{name}'; expected one of: {supported}") from exc
