"""Core type objects for canonical µhLS IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCALAR_TYPE_NAMES = frozenset({"i1", "i8", "i16", "i32", "u8", "u16", "u32"})


@dataclass(frozen=True)
class ScalarType:
    """One supported canonical scalar integer type."""

    name: str

    def __post_init__(self) -> None:
        if self.name not in SCALAR_TYPE_NAMES:
            raise ValueError(f"unsupported scalar type '{self.name}'")

    @property
    def signed(self) -> bool:
        """Return whether the type uses signed arithmetic semantics."""
        return self.name.startswith("i")

    @property
    def width(self) -> int:
        """Return the integer bit width."""
        return int(self.name[1:])

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ArrayType:
    """An explicit canonical array memory object type."""

    element_type: ScalarType

    def __str__(self) -> str:
        return f"{self.element_type}[]"


Type = ScalarType | ArrayType
TypeLike = Type | str


def normalize_scalar_type(type_hint: ScalarType | str) -> ScalarType:
    """Convert a scalar type spelling or object into a canonical ``ScalarType``."""
    if isinstance(type_hint, ScalarType):
        return type_hint
    return ScalarType(str(type_hint).strip())


def coerce_scalar_type(type_hint: Any | None) -> ScalarType | None:
    """Best-effort conversion of a scalar type hint into ``ScalarType``.

    Unlike ``normalize_scalar_type``, this helper is tolerant of non-type inputs
    and duck-typed carrier objects. It returns ``None`` when the input does not
    describe one supported scalar type.
    """
    if type_hint is None:
        return None
    if isinstance(type_hint, ScalarType):
        return type_hint
    if isinstance(type_hint, ArrayType):
        return None

    if isinstance(type_hint, str):
        text = type_hint.strip()
    else:
        text = None
        for attr in ("spelling", "name", "text", "kind"):
            value = getattr(type_hint, attr, None)
            if isinstance(value, str):
                text = value.strip()
                break

    if not text:
        return None

    try:
        return ScalarType(text)
    except ValueError:
        return None


def normalize_type(type_hint: TypeLike | None) -> Type | None:
    """Convert a type spelling or object into a canonical type object."""
    if type_hint is None:
        return None
    if isinstance(type_hint, (ScalarType, ArrayType)):
        return type_hint

    text = str(type_hint).strip()
    if text.endswith("[]"):
        return ArrayType(normalize_scalar_type(text[:-2]))
    return normalize_scalar_type(text)


def type_name(type_hint: TypeLike | None) -> str | None:
    """Return a printable canonical spelling for ``type_hint``."""
    normalized = normalize_type(type_hint)
    return None if normalized is None else str(normalized)
