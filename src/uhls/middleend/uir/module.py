"""Module-level container for canonical µhLS IR."""

from __future__ import annotations

from dataclasses import dataclass, field

from .function import Function


@dataclass(slots=True)
class Module:
    """A named collection of functions and optional external symbols."""

    functions: list[Function] = field(default_factory=list)
    name: str | None = None
    externals: set[str] = field(default_factory=set)

    def function_map(self) -> dict[str, Function]:
        """Return functions indexed by their symbol names."""
        return {function.name: function for function in self.functions}

    def get_function(self, name: str) -> Function | None:
        """Return one function by name if present."""
        return self.function_map().get(name)
