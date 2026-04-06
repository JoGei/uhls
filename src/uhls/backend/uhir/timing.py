"""Timing expression model and parser for symbolic sched µhIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


class TimingExpr:
    """Base class for one symbolic timing expression."""

    @property
    def precedence(self) -> int:
        raise NotImplementedError


TimingAtom: TypeAlias = int | TimingExpr


@dataclass(slots=True, frozen=True)
class TimingVar(TimingExpr):
    name: str

    @property
    def precedence(self) -> int:
        return 4

    def __str__(self) -> str:
        return self.name


@dataclass(slots=True, frozen=True)
class TimingCall(TimingExpr):
    name: str
    args: tuple[TimingAtom, ...]

    @property
    def precedence(self) -> int:
        return 4

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(arg) for arg in self.args)})"


@dataclass(slots=True, frozen=True)
class TimingUnary(TimingExpr):
    op: str
    operand: TimingAtom

    @property
    def precedence(self) -> int:
        return 3

    def __str__(self) -> str:
        return f"{self.op}{_format_timing_atom(self.operand, self.precedence)}"


@dataclass(slots=True, frozen=True)
class TimingBinary(TimingExpr):
    left: TimingAtom
    op: str
    right: TimingAtom

    @property
    def precedence(self) -> int:
        if self.op == "*":
            return 2
        return 1

    def __str__(self) -> str:
        left = _format_timing_atom(self.left, self.precedence, is_right=False, parent_op=self.op)
        right = _format_timing_atom(self.right, self.precedence, is_right=True, parent_op=self.op)
        return f"{left} {self.op} {right}"


def _format_timing_atom(value: TimingAtom, parent_precedence: int, *, is_right: bool = False, parent_op: str | None = None) -> str:
    if isinstance(value, int):
        return str(value)
    rendered = str(value)
    if value.precedence < parent_precedence:
        return f"({rendered})"
    if (
        isinstance(value, TimingBinary)
        and is_right
        and value.precedence == parent_precedence
        and parent_op == "-"
    ):
        return f"({rendered})"
    return rendered


def parse_timing_expr(text: str) -> TimingAtom:
    """Parse one textual timing expression into a structured AST."""
    parser = _TimingExprParser(text)
    return parser.parse()


class _TimingExprParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def parse(self) -> TimingAtom:
        value = self._parse_sum()
        self._skip_spaces()
        if self.index != len(self.text):
            raise ValueError(f"unexpected trailing timing syntax: {self.text[self.index:]!r}")
        return value

    def _parse_sum(self) -> TimingAtom:
        value = self._parse_product()
        while True:
            self._skip_spaces()
            if self._match("+"):
                value = TimingBinary(value, "+", self._parse_product())
                continue
            if self._match("-"):
                value = TimingBinary(value, "-", self._parse_product())
                continue
            return value

    def _parse_product(self) -> TimingAtom:
        value = self._parse_factor()
        while True:
            self._skip_spaces()
            if self._match("*"):
                value = TimingBinary(value, "*", self._parse_factor())
                continue
            return value

    def _parse_factor(self) -> TimingAtom:
        self._skip_spaces()
        if self._match("-"):
            operand = self._parse_factor()
            if isinstance(operand, int):
                return -operand
            return TimingUnary("-", operand)
        if self._match("("):
            value = self._parse_sum()
            self._skip_spaces()
            if not self._match(")"):
                raise ValueError("timing expression is missing ')'")
            return value
        return self._parse_atom()

    def _parse_atom(self) -> TimingAtom:
        self._skip_spaces()
        if self.index >= len(self.text):
            raise ValueError("timing expression is incomplete")
        start = self.index
        char = self.text[self.index]
        if char.isdigit():
            while self.index < len(self.text) and self.text[self.index].isdigit():
                self.index += 1
            return int(self.text[start:self.index])
        if char.isalpha() or char == "_":
            self.index += 1
            while self.index < len(self.text) and (self.text[self.index].isalnum() or self.text[self.index] in "_$"):
                self.index += 1
            name = self.text[start:self.index]
            self._skip_spaces()
            if self._match("("):
                args: list[TimingAtom] = []
                self._skip_spaces()
                if not self._match(")"):
                    while True:
                        args.append(self._parse_sum())
                        self._skip_spaces()
                        if self._match(")"):
                            break
                        if not self._match(","):
                            raise ValueError("timing call is missing ',' or ')'")
                return TimingCall(name, tuple(args))
            return TimingVar(name)
        raise ValueError(f"unexpected timing token {char!r}")

    def _skip_spaces(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def _match(self, token: str) -> bool:
        if self.text.startswith(token, self.index):
            self.index += len(token)
            return True
        return False
