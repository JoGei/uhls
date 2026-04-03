"""Function objects for canonical µhLS IR."""

from __future__ import annotations

from dataclasses import dataclass, field

from .block import Block
from .types import ScalarType, TypeLike, normalize_scalar_type
from .values import Parameter


@dataclass(slots=True)
class Function:
    """One canonical µIR function."""

    name: str
    params: list[Parameter] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    return_type: ScalarType | str = field(default="i32")
    entry: str = "entry"
    local_arrays: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.return_type = normalize_scalar_type(self.return_type)

    def block_map(self) -> dict[str, Block]:
        """Return blocks indexed by label."""
        return {block.label: block for block in self.blocks}
