"""Shared DOT rendering helpers."""

from __future__ import annotations

from collections.abc import Iterable


def indent_lines(lines: Iterable[str], prefix: str) -> list[str]:
    """Prefix each non-empty line."""
    return [f"{prefix}{line}" if line else line for line in lines]


def escape_dot_label(text: str) -> str:
    """Escape one string for a DOT label."""
    return text.replace("\\", "\\\\").replace('"', '\\"')
