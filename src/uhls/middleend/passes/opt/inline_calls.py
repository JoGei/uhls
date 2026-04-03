"""Function inlining for extended-call CFG IR."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from uhls.middleend.uir import (
    ArrayType,
    BinaryOp,
    Block,
    BranchOp,
    CallOp,
    CompareOp,
    CondBranchOp,
    ConstOp,
    Function,
    Literal,
    LoadOp,
    Module,
    Parameter,
    PhiOp,
    PrintOp,
    ReturnOp,
    StoreOp,
    UnaryOp,
    Variable,
    normalize_type,
)
from uhls.middleend.passes.opt.simplify_cfg import simplify_cfg_module
from uhls.middleend.passes.util.pass_manager import PassContext


class InlineError(ValueError):
    """Raised when a function call cannot be inlined safely."""


def inline_calls(
    module: Module,
    context: PassContext | None = None,
    pass_args: tuple[str, ...] = (),
) -> Module:
    """Inline direct calls between module-local µIR functions."""
    requested_callees = tuple(arg.strip() for arg in pass_args if arg.strip())
    requested_callee_set = set(requested_callees)
    if context is not None and requested_callees:
        called_callees = _called_callee_names(module)
        warnings = context.data.setdefault("warnings", [])
        for callee_name in requested_callees:
            if callee_name not in called_callees:
                warnings.append(
                    f"inline_calls: requested callee '{callee_name}' was not found in the translation unit"
                )

    result = deepcopy(module)
    function_map = result.function_map()
    _reject_recursive_cycles(function_map)

    counter = 0
    for function in result.functions:
        if function.name == "main":
            continue
        changed = True
        while changed:
            changed = False
            for block_index, block in enumerate(function.blocks):
                for instruction_index, instruction in enumerate(block.instructions):
                    if not isinstance(instruction, CallOp):
                        continue
                    if requested_callee_set and instruction.callee not in requested_callee_set:
                        continue
                    callee = function_map.get(instruction.callee)
                    if callee is None:
                        raise InlineError(f"call to unknown function '{instruction.callee}'")
                    if callee.name == function.name:
                        raise InlineError(f"recursive call in '{function.name}' cannot be inlined")
                    _inline_call_site(function, block_index, instruction_index, callee, counter)
                    counter += 1
                    changed = True
                    break
                if changed:
                    break

    return simplify_cfg_module(result)


@dataclass(frozen=True)
class _InlineCallsPass:
    name: str = "inline_calls"

    def run(self, ir: Module, context: PassContext, pass_args: tuple[str, ...]) -> Module:
        return inline_calls(ir, context, pass_args)


def InlineCallsPass():
    """Return a pass-manager compatible direct-call inliner pass."""
    return _InlineCallsPass()


def _inline_call_site(
    function: Function,
    block_index: int,
    instruction_index: int,
    callee: Function,
    counter: int,
) -> None:
    block = function.blocks[block_index]
    call = block.instructions[instruction_index]
    if not isinstance(call, CallOp):
        raise InlineError("inline target is not a call instruction")

    if len(call.operands) != len(callee.params):
        raise InlineError(
            f"call to '{callee.name}' expected {len(callee.params)} arguments, got {len(call.operands)}"
        )

    prefix = f"inl_{callee.name}_{counter}"
    continuation = Block(
        label=f"{prefix}_cont",
        instructions=deepcopy(block.instructions[instruction_index + 1 :]),
        terminator=deepcopy(block.terminator),
    )

    block.instructions = block.instructions[:instruction_index]
    block.terminator = BranchOp(f"{prefix}_{callee.entry}")

    name_map: dict[str, str] = {}
    array_map: dict[str, str] = {}
    for param in callee.params:
        if isinstance(normalize_type(param.type), ArrayType):
            continue
        name_map[param.name] = f"{prefix}_{param.name}"
    for local_name, spec in callee.local_arrays.items():
        remapped = f"{prefix}_{local_name}"
        array_map[local_name] = remapped
        function.local_arrays[remapped] = deepcopy(spec)

    for callee_block in callee.blocks:
        for instruction in callee_block.instructions:
            dest = getattr(instruction, "dest", None)
            if dest:
                name_map[dest] = f"{prefix}_{dest}"

    cloned_blocks: list[Block] = []
    for callee_block in callee.blocks:
        cloned_instructions: list[object] = []
        if callee_block.label == callee.entry:
            for param, operand in zip(callee.params, call.operands, strict=True):
                param_type = normalize_type(param.type)
                if isinstance(param_type, ArrayType):
                    array_map[param.name] = _operand_name(operand)
                    continue
                target = name_map[param.name]
                cloned_instructions.append(_materialize_assignment(target, param.type, operand, {}, {}))

        for instruction in callee_block.instructions:
            cloned_instructions.append(_clone_instruction(instruction, name_map, array_map, prefix))

        terminator = _clone_terminator(
            callee_block.terminator,
            name_map,
            array_map,
            prefix,
            call,
            callee.return_type,
            cloned_instructions,
            continuation.label,
        )
        cloned_blocks.append(
            Block(
                label=f"{prefix}_{callee_block.label}",
                instructions=cloned_instructions,
                terminator=terminator,
            )
        )

    function.blocks = [
        *function.blocks[: block_index + 1],
        *cloned_blocks,
        continuation,
        *function.blocks[block_index + 1 :],
    ]


def _clone_instruction(
    instruction: object,
    name_map: dict[str, str],
    array_map: dict[str, str],
    prefix: str,
) -> object:
    if isinstance(instruction, ConstOp):
        return ConstOp(name_map[instruction.dest], instruction.type, instruction.value)
    if isinstance(instruction, UnaryOp):
        return UnaryOp(
            instruction.opcode,
            name_map[instruction.dest],
            instruction.type,
            _rewrite_operand(instruction.value, name_map, array_map),
        )
    if isinstance(instruction, BinaryOp):
        return BinaryOp(
            instruction.opcode,
            name_map[instruction.dest],
            instruction.type,
            _rewrite_operand(instruction.lhs, name_map, array_map),
            _rewrite_operand(instruction.rhs, name_map, array_map),
        )
    if isinstance(instruction, CompareOp):
        return CompareOp(
            instruction.opcode,
            name_map[instruction.dest],
            _rewrite_operand(instruction.lhs, name_map, array_map),
            _rewrite_operand(instruction.rhs, name_map, array_map),
        )
    if isinstance(instruction, LoadOp):
        return LoadOp(
            name_map[instruction.dest],
            instruction.type,
            array_map.get(instruction.array, instruction.array),
            _rewrite_operand(instruction.index, name_map, array_map),
        )
    if isinstance(instruction, StoreOp):
        return StoreOp(
            array_map.get(instruction.array, instruction.array),
            _rewrite_operand(instruction.index, name_map, array_map),
            _rewrite_operand(instruction.value, name_map, array_map),
        )
    if isinstance(instruction, PhiOp):
        return PhiOp(
            name_map[instruction.dest],
            instruction.type,
            [
                (
                    f"{prefix}_{incoming.pred}",
                    _rewrite_operand(incoming.value, name_map, array_map),
                )
                for incoming in instruction.incoming
            ],
        )
    if isinstance(instruction, CallOp):
        return CallOp(
            instruction.callee,
            [_rewrite_operand(operand, name_map, array_map) for operand in instruction.operands],
            dest=None if instruction.dest is None else name_map[instruction.dest],
            type=instruction.type,
        )
    if isinstance(instruction, PrintOp):
        return PrintOp(
            instruction.format,
            [_rewrite_operand(operand, name_map, array_map) for operand in instruction.operands],
        )
    raise InlineError(f"unsupported instruction for inlining: {instruction!r}")


def _clone_terminator(
    terminator: object,
    name_map: dict[str, str],
    array_map: dict[str, str],
    prefix: str,
    call: CallOp,
    return_type: Any,
    instructions: list[object],
    continuation_label: str,
) -> object:
    if isinstance(terminator, BranchOp):
        return BranchOp(f"{prefix}_{terminator.target}")
    if isinstance(terminator, CondBranchOp):
        return CondBranchOp(
            _rewrite_operand(terminator.cond, name_map, array_map),
            f"{prefix}_{terminator.true_target}",
            f"{prefix}_{terminator.false_target}",
        )
    if isinstance(terminator, ReturnOp):
        if call.dest is not None:
            if terminator.value is None:
                raise InlineError("void returns are not supported by the inliner")
            instructions.append(
                _materialize_assignment(
                    call.dest,
                    call.type or return_type,
                    _rewrite_operand(terminator.value, name_map, array_map),
                    {},
                    {},
                )
            )
        return BranchOp(continuation_label)
    raise InlineError(f"unsupported terminator for inlining: {terminator!r}")


def _materialize_assignment(
    target: str,
    target_type: Any,
    operand: object,
    name_map: dict[str, str],
    array_map: dict[str, str],
) -> object:
    rewritten = _rewrite_operand(operand, name_map, array_map)
    if isinstance(rewritten, Literal):
        return ConstOp(target, target_type, rewritten.value)
    return UnaryOp("mov", target, target_type, rewritten)


def _rewrite_operand(operand: object, name_map: dict[str, str], array_map: dict[str, str]) -> object:
    if isinstance(operand, Literal):
        return replace(operand)
    if isinstance(operand, Variable):
        return Variable(name_map.get(operand.name, array_map.get(operand.name, operand.name)), operand.type)
    if isinstance(operand, Parameter):
        return Variable(name_map.get(operand.name, array_map.get(operand.name, operand.name)), operand.type)
    if isinstance(operand, str):
        return name_map.get(operand, array_map.get(operand, operand))
    return deepcopy(operand)


def _operand_name(operand: object) -> str:
    if isinstance(operand, (Variable, Parameter)):
        return operand.name
    if isinstance(operand, str):
        return operand
    raise InlineError(f"array arguments must be symbol names, got {operand!r}")


def _reject_recursive_cycles(function_map: dict[str, Function]) -> None:
    edges: dict[str, set[str]] = {name: set() for name in function_map}
    for function in function_map.values():
        for block in function.blocks:
            for instruction in block.instructions:
                if isinstance(instruction, CallOp) and instruction.callee in function_map:
                    edges[function.name].add(instruction.callee)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise InlineError(f"recursive cycle involving '{name}' cannot be inlined")
        visiting.add(name)
        for target in edges[name]:
            visit(target)
        visiting.remove(name)
        visited.add(name)

    for function_name in function_map:
        visit(function_name)


def _called_callee_names(module: Module) -> set[str]:
    called_callees: set[str] = set()
    for function in module.functions:
        for block in function.blocks:
            for instruction in block.instructions:
                if isinstance(instruction, CallOp):
                    called_callees.add(instruction.callee)
    return called_callees
