"""Text parsing helpers for canonical µhLS IR."""

from __future__ import annotations

import ast as pyast
import re

from uhls.middleend.uir.block import Block
from uhls.middleend.uir.function import Function
from uhls.middleend.uir.module import Module
from uhls.middleend.uir.ops import (
    _BINARY_OPS,
    _COMPARE_OPS,
    _UNARY_OPS,
    BinaryOp,
    BranchOp,
    CallOp,
    CompareOp,
    CondBranchOp,
    ConstOp,
    LoadOp,
    ParamOp,
    PhiOp,
    PrintOp,
    ReturnOp,
    StoreOp,
    UnaryOp,
)
from uhls.middleend.uir.values import IncomingValue, Literal, Parameter

_FUNC_RE = re.compile(r"^func\s+([A-Za-z_]\w*)\((.*)\)\s*->\s*([A-Za-z0-9_\[\]]+)$")
_LOCAL_RE = re.compile(r"^local\s+([A-Za-z_][\w$]*)\[(\d+)\]:([A-Za-z0-9_\[\]]+)$")
_BLOCK_RE = re.compile(r"^block\s+([A-Za-z_]\w*):$")
_ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*):([A-Za-z0-9_\[\]]+)\s*=\s*(.+)$")
_PARAM_RE = re.compile(r"^([A-Za-z_]\w*):([A-Za-z0-9_\[\]]+)$")
_ARRAY_REF_RE = re.compile(r"^([A-Za-z_][\w$]*)\[(.+)\]$")
_CALL_RE = re.compile(r"^call\s+([A-Za-z_]\w*)\((.*)\)$")
_LOWERED_CALL_RE = re.compile(r"^call\s+([A-Za-z_]\w*)\s*,\s*(\d+)$")
_TYPED_LITERAL_RE = re.compile(r"^(-?\d+):(i1|i8|i16|i32|u8|u16|u32)$")
_BARE_INT_RE = re.compile(r"^-?\d+$")


class IRParseError(ValueError):
    """Raised when canonical IR text cannot be parsed."""


def parse_module(text: str) -> Module:
    """Parse canonical IR text into one module."""
    lines = text.splitlines()
    index = 0
    module_name: str | None = None

    while index < len(lines) and not lines[index].strip():
        index += 1

    if index < len(lines) and lines[index].startswith("module "):
        module_name = lines[index][len("module ") :].strip() or None
        index += 1

    functions: list[Function] = []
    while True:
        index = _skip_blank_lines(lines, index)
        if index >= len(lines):
            break
        function, index = _parse_function(lines, index)
        functions.append(function)

    return Module(functions=functions, name=module_name)


def _parse_function(lines: list[str], index: int) -> tuple[Function, int]:
    header = lines[index].strip()
    match = _FUNC_RE.fullmatch(header)
    if match is None:
        raise IRParseError(f"expected function header at line {index + 1}: {lines[index]!r}")

    name, params_text, return_type = match.groups()
    index += 1
    params = _parse_params(params_text)
    local_arrays: dict[str, dict[str, object]] = {}
    blocks: list[Block] = []

    while True:
        index = _skip_blank_lines(lines, index)
        if index >= len(lines):
            break
        local_match = _LOCAL_RE.fullmatch(lines[index].strip())
        if local_match is None:
            break
        local_name, size_text, element_type = local_match.groups()
        local_arrays[local_name] = {"size": int(size_text), "element_type": element_type}
        index += 1

    while True:
        index = _skip_blank_lines(lines, index)
        if index >= len(lines):
            break
        if lines[index].startswith("func "):
            break
        block, index = _parse_block(lines, index)
        blocks.append(block)

    entry = blocks[0].label if blocks else "entry"
    return Function(
        name=name,
        params=params,
        blocks=blocks,
        return_type=return_type,
        entry=entry,
        local_arrays=local_arrays,
    ), index


def _parse_block(lines: list[str], index: int) -> tuple[Block, int]:
    header = lines[index].strip()
    match = _BLOCK_RE.fullmatch(header)
    if match is None:
        raise IRParseError(f"expected block header at line {index + 1}: {lines[index]!r}")

    label = match.group(1)
    index += 1
    instructions: list[object] = []
    terminator = None

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            if index >= len(lines):
                break
            if not lines[index].startswith((" ", "\t")):
                break
            continue
        if not line.startswith((" ", "\t")):
            break

        operation = _parse_operation(stripped, index + 1)
        if getattr(operation, "opcode", None) in {"br", "cbr", "ret"}:
            terminator = operation
        else:
            instructions.append(operation)
        index += 1

    return Block(label=label, instructions=instructions, terminator=terminator), index


def _parse_params(text: str) -> list[Parameter]:
    if not text.strip():
        return []
    params: list[Parameter] = []
    for item in _split_top_level(text):
        match = _PARAM_RE.fullmatch(item.strip())
        if match is None:
            raise IRParseError(f"invalid parameter spelling {item!r}")
        name, type_hint = match.groups()
        params.append(Parameter(name, type_hint))
    return params


def _parse_operation(text: str, line_number: int) -> object:
    if text.startswith("br "):
        return BranchOp(text[3:].strip())
    if text.startswith("cbr "):
        parts = _split_top_level(text[4:])
        if len(parts) != 3:
            raise IRParseError(f"invalid cbr syntax at line {line_number}")
        return CondBranchOp(
            _parse_operand(parts[0]),
            parts[1].strip(),
            parts[2].strip(),
        )
    if text == "ret":
        return ReturnOp()
    if text.startswith("ret "):
        return ReturnOp(_parse_operand(text[4:]))
    if text.startswith("store "):
        ref_text, value_text = _split_once_top_level(text[6:], ",")
        array, index = _parse_array_ref(ref_text.strip(), line_number)
        return StoreOp(array, _parse_operand(index), _parse_operand(value_text))
    if text.startswith("param "):
        parts = _split_top_level(text[6:])
        if len(parts) != 2:
            raise IRParseError(f"invalid param syntax at line {line_number}")
        return ParamOp(int(parts[0].strip()), _parse_operand(parts[1]))
    if text.startswith("print "):
        return _parse_print(text[6:], line_number)
    if text.startswith("call "):
        callee, operands, arg_count = _parse_call(text, line_number)
        return CallOp(callee, operands, arg_count=arg_count)

    match = _ASSIGN_RE.fullmatch(text)
    if match is None:
        raise IRParseError(f"unsupported IR syntax at line {line_number}: {text!r}")

    dest, type_hint, expr = match.groups()
    if expr.startswith("const "):
        operand = _parse_operand(expr[6:])
        if isinstance(operand, Literal):
            return ConstOp(dest, type_hint, operand.value)
        if isinstance(operand, int):
            return ConstOp(dest, type_hint, operand)
        raise IRParseError(f"const expects an integer literal at line {line_number}")
    if expr.startswith("load "):
        array, index = _parse_array_ref(expr[5:].strip(), line_number)
        return LoadOp(dest, type_hint, array, _parse_operand(index))
    if expr.startswith("phi(") and expr.endswith(")"):
        inside = expr[4:-1]
        incoming: list[IncomingValue] = []
        for item in _split_top_level(inside):
            pred_text, value_text = item.split(":", 1)
            incoming.append(IncomingValue(pred_text.strip(), _parse_operand(value_text)))
        return PhiOp(dest, type_hint, incoming)
    if expr.startswith("call "):
        callee, operands, arg_count = _parse_call(expr, line_number)
        return CallOp(callee, operands, dest=dest, type=type_hint, arg_count=arg_count)

    opcode, tail = _split_opcode(expr, line_number)
    if opcode in _UNARY_OPS:
        return UnaryOp(opcode, dest, type_hint, _parse_operand(tail))
    if opcode in _BINARY_OPS:
        lhs, rhs = _parse_binary_operands(tail, line_number)
        return BinaryOp(opcode, dest, type_hint, lhs, rhs)
    if opcode in _COMPARE_OPS:
        lhs, rhs = _parse_binary_operands(tail, line_number)
        return CompareOp(opcode, dest, lhs, rhs)

    raise IRParseError(f"unsupported opcode '{opcode}' at line {line_number}")


def _parse_call(text: str, line_number: int) -> tuple[str, list[object], int | None]:
    match = _CALL_RE.fullmatch(text)
    if match is not None:
        callee, operands_text = match.groups()
        operands = [] if not operands_text.strip() else [_parse_operand(item) for item in _split_top_level(operands_text)]
        return callee, operands, None

    lowered = _LOWERED_CALL_RE.fullmatch(text)
    if lowered is not None:
        callee, arg_count_text = lowered.groups()
        return callee, [], int(arg_count_text)

    raise IRParseError(f"invalid call syntax at line {line_number}")


def _parse_print(text: str, line_number: int) -> PrintOp:
    format_text, rest = _parse_string_prefix(text.strip(), line_number)
    remaining = rest.strip()
    if not remaining:
        return PrintOp(format_text, [])
    if not remaining.startswith(","):
        raise IRParseError(f"invalid print syntax at line {line_number}")
    operands = [_parse_operand(item) for item in _split_top_level(remaining[1:])]
    return PrintOp(format_text, operands)


def _parse_array_ref(text: str, line_number: int) -> tuple[str, str]:
    match = _ARRAY_REF_RE.fullmatch(text)
    if match is None:
        raise IRParseError(f"invalid array reference at line {line_number}: {text!r}")
    return match.group(1), match.group(2).strip()


def _split_opcode(text: str, line_number: int) -> tuple[str, str]:
    parts = text.split(None, 1)
    if len(parts) != 2:
        raise IRParseError(f"missing opcode operands at line {line_number}")
    return parts[0], parts[1]


def _parse_binary_operands(text: str, line_number: int) -> tuple[object, object]:
    parts = _split_top_level(text)
    if len(parts) != 2:
        raise IRParseError(f"binary opcode expects two operands at line {line_number}")
    return _parse_operand(parts[0]), _parse_operand(parts[1])


def _parse_string_prefix(text: str, line_number: int) -> tuple[str, str]:
    if not text or text[0] != '"':
        raise IRParseError(f"expected string literal at line {line_number}")

    escaped = False
    for index in range(1, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            literal_text = text[: index + 1]
            try:
                value = pyast.literal_eval(literal_text)
            except (SyntaxError, ValueError) as exc:
                raise IRParseError(f"invalid string literal at line {line_number}") from exc
            return value, text[index + 1 :]
    raise IRParseError(f"unterminated string literal at line {line_number}")


def _parse_operand(text: str) -> object:
    operand = text.strip()
    typed = _TYPED_LITERAL_RE.fullmatch(operand)
    if typed is not None:
        value, type_hint = typed.groups()
        return Literal(int(value), type_hint)
    if _BARE_INT_RE.fullmatch(operand):
        return int(operand)
    return operand


def _skip_blank_lines(lines: list[str], index: int) -> int:
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def _split_once_top_level(text: str, delimiter: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == delimiter and depth == 0:
            return text[:index], text[index + 1 :]
    raise IRParseError(f"missing delimiter {delimiter!r} in {text!r}")


def _split_top_level(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    items: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(text[start:index].strip())
            start = index + 1
    items.append(text[start:].strip())
    return items
