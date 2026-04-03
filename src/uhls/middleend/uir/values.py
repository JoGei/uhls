"""Value objects used by canonical µhLS IR nodes."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Type, TypeLike, normalize_scalar_type, normalize_type, type_name


@dataclass(frozen=True)
class Parameter:
    """One typed function parameter."""

    name: str
    type: TypeLike

    def __post_init__(self) -> None:
        normalized = normalize_type(self.type)
        if normalized is None:
            raise ValueError(f"unsupported parameter type '{self.type}'")
        object.__setattr__(self, "type", normalized)


@dataclass(frozen=True)
class Variable:
    """A named scalar symbol reference."""

    name: str
    type: TypeLike | None = None

    def __post_init__(self) -> None:
        if self.type is not None:
            object.__setattr__(self, "type", normalize_type(self.type))


@dataclass(frozen=True)
class Literal:
    """A typed integer literal."""

    value: int
    type: ScalarType | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", int(self.value))
        object.__setattr__(self, "type", normalize_scalar_type(self.type))

    def __str__(self) -> str:
        return f"{self.value}:{type_name(self.type)}"


@dataclass(frozen=True)
class IncomingValue:
    """One ``phi`` predecessor/value pair."""

    pred: str
    value: object
