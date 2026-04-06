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
    return simplify_timing_expr(parser.parse())


def simplify_timing_expr(value: TimingAtom) -> TimingAtom:
    """Canonicalize one timing expression using safe timing-domain identities."""
    if isinstance(value, int):
        return value
    if isinstance(value, TimingVar):
        return value
    if isinstance(value, TimingUnary):
        operand = simplify_timing_expr(value.operand)
        if value.op == "-" and isinstance(operand, int):
            return -operand
        if value.op == "-" and isinstance(operand, TimingUnary) and operand.op == "-":
            return operand.operand
        return TimingUnary(value.op, operand)
    if isinstance(value, TimingBinary):
        return _simplify_binary(value)
    if isinstance(value, TimingCall):
        return _simplify_call(value)
    return value


def _simplify_binary(expr: TimingBinary) -> TimingAtom:
    left = simplify_timing_expr(expr.left)
    right = simplify_timing_expr(expr.right)
    if expr.op == "*":
        if isinstance(left, int) and isinstance(right, int):
            return left * right
        if (isinstance(left, int) and left == 0) or (isinstance(right, int) and right == 0):
            return 0
        if isinstance(left, int) and left == 1:
            return right
        if isinstance(right, int) and right == 1:
            return left
        return TimingBinary(left, "*", right)

    terms, constant = _flatten_additive(expr.op, left, right)
    simplified_terms = [term for term in terms if not (isinstance(term, int) and term == 0)]
    if not simplified_terms:
        return constant

    result: TimingAtom | None = None
    for sign, term in simplified_terms:
        current: TimingAtom = term
        if sign < 0:
            if isinstance(term, int):
                current = -term
            else:
                current = TimingUnary("-", term)
        if result is None:
            result = current
            continue
        if isinstance(current, int) and current < 0:
            result = TimingBinary(result, "-", -current)
        elif isinstance(current, TimingUnary) and current.op == "-":
            result = TimingBinary(result, "-", current.operand)
        else:
            result = TimingBinary(result, "+", current)
    assert result is not None
    if constant > 0:
        result = TimingBinary(result, "+", constant)
    elif constant < 0:
        result = TimingBinary(result, "-", -constant)
    return result


def _flatten_additive(op: str, left: TimingAtom, right: TimingAtom) -> tuple[list[tuple[int, TimingAtom]], int]:
    terms: list[tuple[int, TimingAtom]] = []
    constant = 0

    def append(value: TimingAtom, sign: int) -> None:
        nonlocal constant
        if isinstance(value, int):
            constant += sign * value
            return
        if isinstance(value, TimingUnary) and value.op == "-":
            append(value.operand, -sign)
            return
        if isinstance(value, TimingBinary) and value.op in {"+", "-"}:
            sub_terms, sub_const = _flatten_additive(value.op, value.left, value.right)
            for sub_sign, term in sub_terms:
                terms.append((sign * sub_sign, term))
            constant += sign * sub_const
            return
        terms.append((sign, value))

    append(left, 1)
    append(right, 1 if op == "+" else -1)
    combined: dict[str, int] = {}
    term_by_key: dict[str, TimingAtom] = {}
    for sign, term in terms:
        key = str(term)
        combined[key] = combined.get(key, 0) + sign
        term_by_key[key] = term
    ordered_terms: list[tuple[int, TimingAtom]] = []
    for key in sorted(term_by_key):
        coeff = combined[key]
        if coeff == 0:
            continue
        term = term_by_key[key]
        if coeff not in {1, -1}:
            ordered_terms.append((1, TimingBinary(abs(coeff), "*", term)))
            if coeff < 0:
                ordered_terms[-1] = (-1, ordered_terms[-1][1])
        else:
            ordered_terms.append((coeff, term))
    return ordered_terms, constant


def _simplify_call(expr: TimingCall) -> TimingAtom:
    args = tuple(simplify_timing_expr(arg) for arg in expr.args)
    if expr.name != "max":
        return TimingCall(expr.name, args)

    if all(isinstance(arg, int) for arg in args):
        return max(arg for arg in args)

    flattened: list[TimingAtom] = []
    for arg in args:
        if isinstance(arg, TimingCall) and arg.name == "max":
            flattened.extend(arg.args)
        else:
            flattened.append(arg)

    deduped: list[TimingAtom] = []
    seen: set[str] = set()
    for arg in flattened:
        key = str(arg)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(arg)

    non_zero = [arg for arg in deduped if not (arg == 0 and any(_is_non_negative(other) for other in deduped if other != arg))]
    if not non_zero:
        return 0
    affine_max = _simplify_affine_max(non_zero)
    if affine_max is not None:
        return affine_max
    if len(non_zero) == 1:
        return non_zero[0]
    return TimingCall("max", tuple(non_zero))


def _simplify_affine_max(args: list[TimingAtom]) -> TimingAtom | None:
    signature0, constant0 = _additive_signature(args[0])
    if signature0 is None:
        return None
    constants = [constant0]
    for arg in args[1:]:
        signature, constant = _additive_signature(arg)
        if signature != signature0 or signature is None:
            return None
        constants.append(constant)
    base = _rebuild_from_signature(signature0)
    max_constant = max(constants)
    if max_constant == 0:
        return base
    return simplify_timing_expr(TimingBinary(base, "+", max_constant))


def _additive_signature(value: TimingAtom) -> tuple[tuple[tuple[int, str], ...], int] | tuple[None, int]:
    if isinstance(value, int):
        return tuple(), value
    terms, constant = _flatten_additive("+", value, 0)
    signature: list[tuple[int, str]] = []
    for sign, term in terms:
        if isinstance(term, int):
            return None, constant
        signature.append((sign, str(term)))
    return tuple(signature), constant


def _rebuild_from_signature(signature: tuple[tuple[int, str], ...]) -> TimingAtom:
    if not signature:
        return 0
    result: TimingAtom | None = None
    for sign, text in signature:
        term = parse_timing_expr(text)
        current: TimingAtom = term if sign > 0 else TimingUnary("-", term) if not isinstance(term, int) else -term
        if result is None:
            result = current
            continue
        if isinstance(current, int) and current < 0:
            result = TimingBinary(result, "-", -current)
        elif isinstance(current, TimingUnary) and current.op == "-":
            result = TimingBinary(result, "-", current.operand)
        else:
            result = TimingBinary(result, "+", current)
    assert result is not None
    return simplify_timing_expr(result)


def _is_non_negative(value: TimingAtom) -> bool:
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, TimingVar):
        return True
    if isinstance(value, TimingUnary):
        return False
    if isinstance(value, TimingCall):
        return value.name == "max" and all(_is_non_negative(arg) for arg in value.args)
    if isinstance(value, TimingBinary):
        if value.op == "+":
            return _is_non_negative(value.left) and _is_non_negative(value.right)
        if value.op == "*":
            if isinstance(value.left, int):
                return value.left >= 0 and _is_non_negative(value.right)
            if isinstance(value.right, int):
                return value.right >= 0 and _is_non_negative(value.left)
            return _is_non_negative(value.left) and _is_non_negative(value.right)
        if value.op == "-":
            if isinstance(value.right, int):
                return _is_non_negative(value.left) and value.right == 0
            return False
    return False


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
