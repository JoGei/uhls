"""Basic block objects for canonical µhLS IR."""

from __future__ import annotations

from dataclasses import dataclass, field

from .ops import Instruction, Terminator


@dataclass(slots=True)
class Block:
    """One explicit basic block with a separate terminator."""

    label: str
    instructions: list[Instruction] = field(default_factory=list)
    terminator: Terminator | None = None

    def all_instructions(self) -> list[object]:
        """Return the block body followed by its terminator when present."""
        if self.terminator is None:
            return list(self.instructions)
        return [*self.instructions, self.terminator]
