"""DOT rendering helpers for bind-stage conflict/coloring views."""

from __future__ import annotations

from uhls.backend.uhir.model import UHIRDesign

from .analysis import bind_dump_to_dot


def binding_to_dot(design: UHIRDesign, compact: bool = False) -> str:
    """Render one bind-stage design as operation-conflict DOT graphs."""
    return bind_dump_to_dot(design, ("conflict",), compact=compact)
