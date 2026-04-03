"""Dead-code elimination for canonical µhLS IR."""

from __future__ import annotations

from copy import deepcopy

from uhls.middleend.uir import (
    BinaryOp,
    Block,
    CallOp,
    CompareOp,
    CondBranchOp,
    Function,
    Literal,
    LoadOp,
    Module,
    ParamOp,
    Parameter,
    PhiOp,
    PrintOp,
    ReturnOp,
    StoreOp,
    UnaryOp,
    Variable,
)
from uhls.middleend.passes.analyze import liveness
from uhls.middleend.passes.util.pass_manager import function_or_module_pass


def dce_function(function: Function) -> Function:
    """Eliminate dead scalar instructions from one function."""
    result = deepcopy(function)
    changed = True
    while changed:
        changed = False
        info = liveness(result)
        rewritten_blocks: list[Block] = []
        for block in result.blocks:
            live = set(info.live_out[block.label])
            live.update(_terminator_uses(block.terminator))
            kept_reversed: list[object] = []

            for instruction in reversed(block.instructions):
                dest = getattr(instruction, "dest", None)
                keep = _has_side_effects(instruction) or not isinstance(dest, str) or dest in live
                if not keep:
                    changed = True
                    continue
                kept_reversed.append(instruction)
                if isinstance(dest, str):
                    live.discard(dest)
                live.update(_instruction_uses(instruction))

            rewritten_blocks.append(
                Block(
                    label=block.label,
                    instructions=list(reversed(kept_reversed)),
                    terminator=block.terminator,
                )
            )

        result = Function(
            name=result.name,
            params=deepcopy(result.params),
            blocks=rewritten_blocks,
            return_type=result.return_type,
            entry=result.entry,
            local_arrays=deepcopy(result.local_arrays),
        )
    return result


def dce_module(module: Module) -> Module:
    """Eliminate dead scalar instructions from every function in one module."""
    result = deepcopy(module)
    result.functions = [dce_function(function) for function in result.functions]
    return result


def DCEPass():
    """Return a pass-manager compatible DCE pass wrapper."""
    return function_or_module_pass("dce", dce_function, dce_module)


def _has_side_effects(instruction: object) -> bool:
    return isinstance(instruction, (CallOp, PrintOp, StoreOp, ParamOp))


def _instruction_uses(instruction: object) -> set[str]:
    if isinstance(instruction, PhiOp):
        return set()
    if isinstance(instruction, UnaryOp):
        return _operand_names(instruction.value)
    if isinstance(instruction, (BinaryOp, CompareOp)):
        return _operand_names(instruction.lhs) | _operand_names(instruction.rhs)
    if isinstance(instruction, LoadOp):
        return _operand_names(instruction.index)
    if isinstance(instruction, StoreOp):
        return _operand_names(instruction.index) | _operand_names(instruction.value)
    if isinstance(instruction, CallOp):
        used: set[str] = set()
        for operand in instruction.operands:
            used |= _operand_names(operand)
        return used
    if isinstance(instruction, PrintOp):
        used: set[str] = set()
        for operand in instruction.operands:
            used |= _operand_names(operand)
        return used
    if isinstance(instruction, ParamOp):
        return _operand_names(instruction.value)
    return set()


def _terminator_uses(terminator: object) -> set[str]:
    if isinstance(terminator, CondBranchOp):
        return _operand_names(terminator.cond)
    if isinstance(terminator, ReturnOp) and terminator.value is not None:
        return _operand_names(terminator.value)
    return set()


def _operand_names(operand: object) -> set[str]:
    if isinstance(operand, Literal):
        return set()
    if isinstance(operand, Variable):
        return {operand.name}
    if isinstance(operand, Parameter):
        return {operand.name}
    if isinstance(operand, str):
        return {operand}
    return set()
