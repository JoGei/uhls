"""Constant propagation for canonical µhLS IR."""

from __future__ import annotations

from copy import deepcopy

from uhls.interpreter.eval import eval_binary, eval_compare, eval_unary, truthy
from uhls.middleend.uir import (
    BinaryOp,
    Block,
    BranchOp,
    CallOp,
    CompareOp,
    CondBranchOp,
    ConstOp,
    Function,
    IncomingValue,
    Literal,
    LoadOp,
    Module,
    ParamOp,
    PhiOp,
    PrintOp,
    ReturnOp,
    StoreOp,
    UnaryOp,
    Variable,
)
from uhls.middleend.passes.util.pass_manager import function_or_module_pass


def const_prop_function(function: Function) -> Function:
    """Propagate constants through one function and fold pure constant expressions."""
    result = deepcopy(function)
    changed = True

    while changed:
        changed = False
        constants = _discover_constants(result)
        rewritten_blocks: list[Block] = []

        for block in result.blocks:
            new_instructions: list[object] = []
            for instruction in block.instructions:
                rewritten = _rewrite_instruction(instruction, constants)
                if rewritten != instruction:
                    changed = True
                new_instructions.append(rewritten)

            rewritten_terminator = _rewrite_terminator(block.terminator, constants)
            if rewritten_terminator != block.terminator:
                changed = True

            rewritten_blocks.append(
                Block(
                    label=block.label,
                    instructions=new_instructions,
                    terminator=rewritten_terminator,
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


def const_prop_module(module: Module) -> Module:
    """Propagate constants through every function in one module."""
    result = deepcopy(module)
    result.functions = [const_prop_function(function) for function in result.functions]
    return result


def ConstPropPass():
    """Return a pass-manager compatible constant-propagation pass wrapper."""
    return function_or_module_pass("const_prop", const_prop_function, const_prop_module)


def _discover_constants(function: Function) -> dict[str, Literal]:
    constants: dict[str, Literal] = {}
    changed = True

    while changed:
        changed = False
        for block in function.blocks:
            for instruction in block.instructions:
                dest = getattr(instruction, "dest", None)
                if not isinstance(dest, str):
                    continue
                literal = _constant_result(instruction, constants)
                if literal is None or constants.get(dest) == literal:
                    continue
                constants[dest] = literal
                changed = True

    return constants


def _constant_result(instruction: object, constants: dict[str, Literal]) -> Literal | None:
    if isinstance(instruction, ConstOp):
        return Literal(instruction.value, instruction.type)

    if isinstance(instruction, UnaryOp):
        operand = _literal_operand(instruction.value, constants)
        if operand is None:
            return None
        try:
            value = eval_unary(instruction.opcode, operand.value, instruction.type)
        except (ValueError, ZeroDivisionError):
            return None
        return Literal(value, instruction.type)

    if isinstance(instruction, BinaryOp):
        lhs = _literal_operand(instruction.lhs, constants)
        rhs = _literal_operand(instruction.rhs, constants)
        if lhs is None or rhs is None:
            return None
        try:
            value = eval_binary(instruction.opcode, lhs.value, rhs.value, instruction.type)
        except (ValueError, ZeroDivisionError):
            return None
        return Literal(value, instruction.type)

    if isinstance(instruction, CompareOp):
        lhs = _literal_operand(instruction.lhs, constants)
        rhs = _literal_operand(instruction.rhs, constants)
        if lhs is None or rhs is None:
            return None
        try:
            value = eval_compare(instruction.opcode, lhs.value, rhs.value)
        except ValueError:
            return None
        return Literal(value, instruction.type)

    if isinstance(instruction, PhiOp):
        resolved = [_literal_operand(incoming.value, constants) for incoming in instruction.incoming]
        if any(item is None for item in resolved):
            return None
        values = [item.value for item in resolved if item is not None]
        if not values or any(value != values[0] for value in values[1:]):
            return None
        return Literal(values[0], instruction.type)

    return None


def _rewrite_instruction(instruction: object, constants: dict[str, Literal]) -> object:
    folded = _constant_result(instruction, constants)
    if isinstance(instruction, (ConstOp, UnaryOp, BinaryOp, CompareOp, PhiOp)) and folded is not None:
        return ConstOp(getattr(instruction, "dest"), getattr(instruction, "type"), folded.value)

    if isinstance(instruction, UnaryOp):
        return UnaryOp(
            instruction.opcode,
            instruction.dest,
            instruction.type,
            _rewrite_operand(instruction.value, constants),
        )
    if isinstance(instruction, BinaryOp):
        return BinaryOp(
            instruction.opcode,
            instruction.dest,
            instruction.type,
            _rewrite_operand(instruction.lhs, constants),
            _rewrite_operand(instruction.rhs, constants),
        )
    if isinstance(instruction, CompareOp):
        return CompareOp(
            instruction.opcode,
            instruction.dest,
            _rewrite_operand(instruction.lhs, constants),
            _rewrite_operand(instruction.rhs, constants),
        )
    if isinstance(instruction, LoadOp):
        return LoadOp(
            instruction.dest,
            instruction.type,
            instruction.array,
            _rewrite_operand(instruction.index, constants),
        )
    if isinstance(instruction, StoreOp):
        return StoreOp(
            instruction.array,
            _rewrite_operand(instruction.index, constants),
            _rewrite_operand(instruction.value, constants),
        )
    if isinstance(instruction, PhiOp):
        return PhiOp(
            instruction.dest,
            instruction.type,
            [
                IncomingValue(incoming.pred, _rewrite_operand(incoming.value, constants))
                for incoming in instruction.incoming
            ],
        )
    if isinstance(instruction, CallOp):
        return CallOp(
            instruction.callee,
            [_rewrite_operand(operand, constants) for operand in instruction.operands],
            dest=instruction.dest,
            type=instruction.type,
            arg_count=instruction.arg_count,
        )
    if isinstance(instruction, PrintOp):
        return PrintOp(
            instruction.format,
            [_rewrite_operand(operand, constants) for operand in instruction.operands],
        )
    if isinstance(instruction, ParamOp):
        return ParamOp(instruction.index, _rewrite_operand(instruction.value, constants))

    return deepcopy(instruction)


def _rewrite_terminator(terminator: object, constants: dict[str, Literal]) -> object:
    if isinstance(terminator, BranchOp):
        return deepcopy(terminator)
    if isinstance(terminator, CondBranchOp):
        cond = _rewrite_operand(terminator.cond, constants)
        literal = _literal_operand(cond, constants)
        if literal is not None:
            return BranchOp(terminator.true_target if truthy(literal.value) else terminator.false_target)
        return CondBranchOp(cond, terminator.true_target, terminator.false_target)
    if isinstance(terminator, ReturnOp):
        if terminator.value is None:
            return ReturnOp()
        return ReturnOp(_rewrite_operand(terminator.value, constants))
    return deepcopy(terminator)


def _rewrite_operand(operand: object, constants: dict[str, Literal]) -> object:
    literal = _propagated_literal(operand, constants)
    if literal is not None:
        return literal
    return deepcopy(operand)


def _literal_operand(operand: object, constants: dict[str, Literal]) -> Literal | None:
    if isinstance(operand, Literal):
        return operand
    if isinstance(operand, bool):
        return Literal(1 if operand else 0, "i1")
    if isinstance(operand, int):
        return Literal(operand, "i32")
    if isinstance(operand, Variable):
        return constants.get(operand.name)
    if isinstance(operand, str):
        return constants.get(operand)
    return None


def _propagated_literal(operand: object, constants: dict[str, Literal]) -> Literal | None:
    if isinstance(operand, Variable):
        return constants.get(operand.name)
    if isinstance(operand, str):
        return constants.get(operand)
    return None
