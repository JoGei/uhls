"""Common-subexpression elimination for canonical µhLS IR."""

from __future__ import annotations

from copy import deepcopy

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

_COMMUTATIVE_BINARY_OPS = frozenset({"add", "mul", "and", "or", "xor"})
_COMMUTATIVE_COMPARE_OPS = frozenset({"eq", "ne"})


def cse_function(function: Function) -> Function:
    """Eliminate repeated pure scalar computations within basic blocks."""
    result = deepcopy(function)
    changed = True
    while changed:
        changed = False
        replacements: dict[str, str] = {}
        rewritten_blocks: list[Block] = []

        for block in result.blocks:
            expr_to_dest: dict[tuple[object, ...], str] = {}
            new_instructions: list[object] = []

            for instruction in block.instructions:
                rewritten = _rewrite_instruction_operands(instruction, replacements)
                key = _expression_key(rewritten)
                dest = getattr(rewritten, "dest", None)
                if key is not None and isinstance(dest, str):
                    existing = expr_to_dest.get(key)
                    if existing is not None:
                        replacements[dest] = existing
                        changed = True
                        continue
                    expr_to_dest[key] = dest
                new_instructions.append(rewritten)

            rewritten_blocks.append(
                Block(
                    label=block.label,
                    instructions=new_instructions,
                    terminator=_rewrite_terminator_operands(block.terminator, replacements),
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


def cse_module(module: Module) -> Module:
    """Eliminate repeated pure scalar computations from every function in one module."""
    result = deepcopy(module)
    result.functions = [cse_function(function) for function in result.functions]
    return result


def CSEPass():
    """Return a pass-manager compatible CSE pass wrapper."""
    return function_or_module_pass("cse", cse_function, cse_module)


def _expression_key(instruction: object) -> tuple[object, ...] | None:
    if isinstance(instruction, ConstOp):
        return ("const", _type_key(instruction.type), instruction.value)
    if isinstance(instruction, UnaryOp):
        return ("unary", instruction.opcode, _type_key(instruction.type), _operand_key(instruction.value))
    if isinstance(instruction, BinaryOp):
        lhs = _operand_key(instruction.lhs)
        rhs = _operand_key(instruction.rhs)
        if instruction.opcode in _COMMUTATIVE_BINARY_OPS and rhs < lhs:
            lhs, rhs = rhs, lhs
        return ("binary", instruction.opcode, _type_key(instruction.type), lhs, rhs)
    if isinstance(instruction, CompareOp):
        lhs = _operand_key(instruction.lhs)
        rhs = _operand_key(instruction.rhs)
        if instruction.opcode in _COMMUTATIVE_COMPARE_OPS and rhs < lhs:
            lhs, rhs = rhs, lhs
        return ("compare", instruction.opcode, lhs, rhs)
    return None


def _rewrite_instruction_operands(instruction: object, replacements: dict[str, str]) -> object:
    if isinstance(instruction, ConstOp):
        return deepcopy(instruction)
    if isinstance(instruction, UnaryOp):
        return UnaryOp(
            instruction.opcode,
            instruction.dest,
            instruction.type,
            _rewrite_operand(instruction.value, replacements),
        )
    if isinstance(instruction, BinaryOp):
        return BinaryOp(
            instruction.opcode,
            instruction.dest,
            instruction.type,
            _rewrite_operand(instruction.lhs, replacements),
            _rewrite_operand(instruction.rhs, replacements),
        )
    if isinstance(instruction, CompareOp):
        return CompareOp(
            instruction.opcode,
            instruction.dest,
            _rewrite_operand(instruction.lhs, replacements),
            _rewrite_operand(instruction.rhs, replacements),
        )
    if isinstance(instruction, LoadOp):
        return LoadOp(
            instruction.dest,
            instruction.type,
            instruction.array,
            _rewrite_operand(instruction.index, replacements),
        )
    if isinstance(instruction, StoreOp):
        return StoreOp(
            instruction.array,
            _rewrite_operand(instruction.index, replacements),
            _rewrite_operand(instruction.value, replacements),
        )
    if isinstance(instruction, PhiOp):
        return PhiOp(
            instruction.dest,
            instruction.type,
            [
                IncomingValue(incoming.pred, _rewrite_operand(incoming.value, replacements))
                for incoming in instruction.incoming
            ],
        )
    if isinstance(instruction, CallOp):
        return CallOp(
            instruction.callee,
            [_rewrite_operand(operand, replacements) for operand in instruction.operands],
            dest=instruction.dest,
            type=instruction.type,
            arg_count=instruction.arg_count,
        )
    if isinstance(instruction, PrintOp):
        return PrintOp(instruction.format, [_rewrite_operand(operand, replacements) for operand in instruction.operands])
    if isinstance(instruction, ParamOp):
        return ParamOp(instruction.index, _rewrite_operand(instruction.value, replacements))
    return deepcopy(instruction)


def _rewrite_terminator_operands(terminator: object, replacements: dict[str, str]) -> object:
    if isinstance(terminator, BranchOp):
        return deepcopy(terminator)
    if isinstance(terminator, CondBranchOp):
        return CondBranchOp(
            _rewrite_operand(terminator.cond, replacements),
            terminator.true_target,
            terminator.false_target,
        )
    if isinstance(terminator, ReturnOp):
        if terminator.value is None:
            return ReturnOp()
        return ReturnOp(_rewrite_operand(terminator.value, replacements))
    return deepcopy(terminator)


def _rewrite_operand(operand: object, replacements: dict[str, str]) -> object:
    if isinstance(operand, Variable):
        return Variable(_rewrite_name(operand.name, replacements), operand.type)
    if isinstance(operand, str):
        return _rewrite_name(operand, replacements)
    return operand


def _rewrite_name(name: str, replacements: dict[str, str]) -> str:
    current = name
    while current in replacements:
        current = replacements[current]
    return current


def _operand_key(operand: object) -> tuple[object, ...]:
    if isinstance(operand, Literal):
        type_name = getattr(operand.type, "name", str(operand.type))
        return ("literal", type_name, operand.value)
    if isinstance(operand, Variable):
        return ("var", operand.name)
    if isinstance(operand, str):
        return ("name", operand)
    if isinstance(operand, int):
        return ("int", operand)
    return ("other", repr(operand))


def _type_key(type_hint: object) -> str:
    return str(getattr(type_hint, "name", type_hint))
