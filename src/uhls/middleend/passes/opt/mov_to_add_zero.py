"""Rewrite mov aliases into explicit add-with-zero operations."""

from __future__ import annotations

from copy import deepcopy

from uhls.middleend.passes.util.pass_manager import function_or_module_pass
from uhls.middleend.uir import BinaryOp, Block, Function, Literal, Module, UnaryOp


def mov_to_add_zero_function(function: Function) -> Function:
    """Rewrite ``mov`` instructions into ``add value, 0:<type>``."""
    rewritten_blocks: list[Block] = []
    for block in function.blocks:
        rewritten_instructions: list[object] = []
        for instruction in block.instructions:
            if isinstance(instruction, UnaryOp) and instruction.opcode == "mov":
                rewritten_instructions.append(
                    BinaryOp(
                        "add",
                        instruction.dest,
                        instruction.type,
                        deepcopy(instruction.value),
                        Literal(0, deepcopy(instruction.type)),
                    )
                )
                continue
            rewritten_instructions.append(deepcopy(instruction))
        rewritten_blocks.append(
            Block(
                label=block.label,
                instructions=rewritten_instructions,
                terminator=deepcopy(block.terminator),
            )
        )
    return Function(
        name=function.name,
        params=deepcopy(function.params),
        blocks=rewritten_blocks,
        return_type=function.return_type,
        entry=function.entry,
        local_arrays=deepcopy(function.local_arrays),
    )


def mov_to_add_zero_module(module: Module) -> Module:
    """Rewrite ``mov`` instructions throughout one module."""
    result = deepcopy(module)
    result.functions = [mov_to_add_zero_function(function) for function in result.functions]
    return result


def MovToAddZeroPass():
    """Return a pass-manager compatible ``mov`` rewrite pass wrapper."""
    return function_or_module_pass("mov_to_add_zero", mov_to_add_zero_function, mov_to_add_zero_module)
