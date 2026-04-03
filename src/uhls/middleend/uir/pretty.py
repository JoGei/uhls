"""Pretty-print helpers for canonical µhLS IR."""

from __future__ import annotations

import json
from typing import Any

from .block import Block
from .function import Function
from .module import Module
from .ops import (
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
from .types import type_name
from .values import IncomingValue, Literal, Parameter, Variable


def format_operand(operand: Any) -> str:
    """Render one operand using canonical IR syntax."""
    if isinstance(operand, Literal):
        return str(operand)
    if isinstance(operand, (Parameter, Variable)):
        return operand.name
    if isinstance(operand, bool):
        return "1:i1" if operand else "0:i1"
    return str(operand)


def format_instruction(instruction: Any) -> str:
    """Render one non-block IR operation."""
    if isinstance(instruction, ConstOp):
        return (
            f"{instruction.dest}:{type_name(instruction.type)} = const "
            f"{instruction.value}:{type_name(instruction.type)}"
        )
    if isinstance(instruction, UnaryOp):
        return (
            f"{instruction.dest}:{type_name(instruction.type)} = {instruction.opcode} "
            f"{format_operand(instruction.value)}"
        )
    if isinstance(instruction, (BinaryOp, CompareOp)):
        return (
            f"{instruction.dest}:{type_name(instruction.type)} = {instruction.opcode} "
            f"{format_operand(instruction.lhs)}, {format_operand(instruction.rhs)}"
        )
    if isinstance(instruction, LoadOp):
        return (
            f"{instruction.dest}:{type_name(instruction.type)} = load "
            f"{instruction.array}[{format_operand(instruction.index)}]"
        )
    if isinstance(instruction, StoreOp):
        return f"store {instruction.array}[{format_operand(instruction.index)}], {format_operand(instruction.value)}"
    if isinstance(instruction, PhiOp):
        incoming = ", ".join(
            f"{item.pred}: {format_operand(item.value)}"
            for item in instruction.incoming
        )
        return f"{instruction.dest}:{type_name(instruction.type)} = phi({incoming})"
    if isinstance(instruction, CallOp):
        if instruction.arg_count is not None:
            call = f"call {instruction.callee}, {instruction.arg_count}"
        else:
            operands = ", ".join(format_operand(operand) for operand in instruction.operands)
            call = f"call {instruction.callee}({operands})"
        if instruction.dest is None:
            return call
        return f"{instruction.dest}:{type_name(instruction.type)} = {call}"
    if isinstance(instruction, PrintOp):
        operands = ", ".join(format_operand(operand) for operand in instruction.operands)
        head = f'print {json.dumps(instruction.format)}'
        return head if not operands else f"{head}, {operands}"
    if isinstance(instruction, ParamOp):
        return f"param {instruction.index}, {format_operand(instruction.value)}"
    if isinstance(instruction, BranchOp):
        return f"br {instruction.target}"
    if isinstance(instruction, CondBranchOp):
        return (
            f"cbr {format_operand(instruction.cond)}, "
            f"{instruction.true_target}, {instruction.false_target}"
        )
    if isinstance(instruction, ReturnOp):
        if instruction.value is None:
            return "ret"
        return f"ret {format_operand(instruction.value)}"
    raise TypeError(f"cannot pretty-print instruction {instruction!r}")


def format_block(block: Block) -> str:
    """Render one block."""
    lines = [f"block {block.label}:"]
    for instruction in block.instructions:
        lines.append(f"    {format_instruction(instruction)}")
    if block.terminator is not None:
        lines.append(f"    {format_instruction(block.terminator)}")
    return "\n".join(lines)


def format_function(function: Function) -> str:
    """Render one function."""
    params = ", ".join(
        f"{parameter.name}:{type_name(parameter.type)}"
        for parameter in function.params
    )
    head = f"func {function.name}({params}) -> {type_name(function.return_type)}"
    locals_text = "\n".join(
        f"local {name}[{spec['size']}]:{type_name(spec['element_type'])}"
        for name, spec in sorted(function.local_arrays.items())
    )
    blocks = "\n\n".join(format_block(block) for block in function.blocks)
    sections = [head]
    if locals_text:
        sections.append(locals_text)
    if blocks:
        sections.append(blocks)
    return "\n\n".join(sections)


def format_module(module: Module) -> str:
    """Render one module."""
    body = "\n\n".join(format_function(function) for function in module.functions)
    if module.name:
        if body:
            return f"module {module.name}\n\n{body}"
        return f"module {module.name}"
    return body


def pretty(value: Module | Function | Block) -> str:
    """Render a module, function, or block."""
    if isinstance(value, Module):
        return format_module(value)
    if isinstance(value, Function):
        return format_function(value)
    if isinstance(value, Block):
        return format_block(value)
    raise TypeError(f"cannot pretty-print {value!r}")
