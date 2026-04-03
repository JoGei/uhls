"""Construct canonical µIR from plain lowered frontend IR."""

from __future__ import annotations

from copy import deepcopy
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
    IncomingValue,
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
    verify_function,
    verify_module,
)
from uhls.middleend.passes.analyze import compute_dominators, control_flow, liveness
from uhls.middleend.passes.opt import simplify_cfg_function


class UIRConstructionError(ValueError):
    """Raised when lowered frontend IR cannot be converted to canonical µIR."""


def to_uir_function(function: Function) -> Function:
    """Convert one lowered frontend function into canonical scalar-SSA µIR."""
    working = simplify_cfg_function(deepcopy(function))
    _ensure_no_preexisting_phi(working)

    block_map = {block.label: block for block in working.blocks}
    cfg = control_flow(working)
    succs = cfg.successors
    dom_info = compute_dominators(working)
    dom_tree = dom_info.tree
    frontier = dom_info.frontiers

    symbol_types = _scalar_symbol_types(working)
    defsites = _definition_sites(working, symbol_types)
    live_in = liveness(working).live_in
    phi_blocks = _phi_blocks(symbol_types, defsites, frontier, live_in)

    phi_base_names: dict[int, str] = {}
    for block in working.blocks:
        insertions: list[PhiOp] = []
        for name in sorted(phi_blocks.get(block.label, set())):
            phi = PhiOp(name, symbol_types[name], [])
            phi_base_names[id(phi)] = name
            insertions.append(phi)
        block.instructions = insertions + block.instructions

    counters = {name: 0 for name in symbol_types}
    stacks = {name: [] for name in symbol_types}
    renamed_params: list[Parameter] = []
    for param in working.params:
        param_type = normalize_type(param.type)
        if isinstance(param_type, ArrayType):
            renamed_params.append(param)
            continue
        versioned = _new_name(param.name, counters, stacks)
        renamed_params.append(Parameter(versioned, param.type))
    working.params = renamed_params

    _rename_block(
        block_map[working.entry],
        block_map,
        succs,
        dom_tree,
        stacks,
        counters,
        symbol_types,
        phi_base_names,
    )

    verify_function(working, require_ssa=True, allow_calls=True)
    return working


def to_uir_module(module: Module) -> Module:
    """Convert every lowered frontend function in a module into canonical µIR."""
    result = deepcopy(module)
    result.functions = [to_uir_function(function) for function in result.functions]
    verify_module(result, require_ssa=True, allow_calls=True)
    return result


def _ensure_no_preexisting_phi(function: Function) -> None:
    for block in function.blocks:
        for instruction in block.instructions:
            if isinstance(instruction, PhiOp):
                raise UIRConstructionError(
                    f"function '{function.name}' already contains phi nodes and does not look like lowered frontend IR"
                )


def _scalar_symbol_types(function: Function) -> dict[str, Any]:
    symbol_types: dict[str, Any] = {}
    for param in function.params:
        param_type = normalize_type(param.type)
        if isinstance(param_type, ArrayType):
            continue
        symbol_types[param.name] = param.type
    for block in function.blocks:
        for instruction in block.instructions:
            dest = getattr(instruction, "dest", None)
            instruction_type = getattr(instruction, "type", None)
            if dest is None or instruction_type is None:
                continue
            symbol_types[dest] = instruction_type
    return symbol_types


def _definition_sites(function: Function, symbol_types: dict[str, Any]) -> dict[str, set[str]]:
    sites = {name: set() for name in symbol_types}
    for block in function.blocks:
        for instruction in block.instructions:
            dest = getattr(instruction, "dest", None)
            if dest in sites:
                sites[dest].add(block.label)
    return sites


def _phi_blocks(
    symbol_types: dict[str, Any],
    defsites: dict[str, set[str]],
    frontier: dict[str, set[str]],
    live_in: dict[str, set[str]],
) -> dict[str, set[str]]:
    inserted: dict[str, set[str]] = {}
    for name in symbol_types:
        work = list(defsites.get(name, set()))
        seen = set(work)
        while work:
            block = work.pop()
            for frontier_block in frontier.get(block, set()):
                if name not in live_in.get(frontier_block, set()):
                    continue
                bucket = inserted.setdefault(frontier_block, set())
                if name in bucket:
                    continue
                bucket.add(name)
                if frontier_block not in seen:
                    seen.add(frontier_block)
                    work.append(frontier_block)
    return inserted


def _rename_block(
    block: Block,
    block_map: dict[str, Block],
    succs: dict[str, set[str]],
    dom_tree: dict[str, list[str]],
    stacks: dict[str, list[str]],
    counters: dict[str, int],
    symbol_types: dict[str, Any],
    phi_base_names: dict[int, str],
) -> None:
    pushed: list[str] = []

    for instruction in block.instructions:
        if not isinstance(instruction, PhiOp):
            break
        base = phi_base_names[id(instruction)]
        instruction.dest = _new_name(base, counters, stacks)
        pushed.append(base)

    for instruction in block.instructions:
        _rename_instruction_operands(instruction, stacks)
        dest = getattr(instruction, "dest", None)
        if dest is None or isinstance(instruction, PhiOp):
            continue
        base = _base_name(dest)
        if base not in symbol_types:
            continue
        instruction.dest = _new_name(base, counters, stacks)
        pushed.append(base)

    _rename_terminator(block.terminator, stacks)

    for successor_label in succs.get(block.label, set()):
        successor = block_map[successor_label]
        for instruction in successor.instructions:
            if not isinstance(instruction, PhiOp):
                break
            base = phi_base_names[id(instruction)]
            current = _current_name(base, stacks)
            instruction.incoming.append(IncomingValue(block.label, current))

    for child_label in dom_tree.get(block.label, []):
        _rename_block(
            block_map[child_label],
            block_map,
            succs,
            dom_tree,
            stacks,
            counters,
            symbol_types,
            phi_base_names,
        )

    for base in reversed(pushed):
        stacks[base].pop()


def _rename_instruction_operands(instruction: object, stacks: dict[str, list[str]]) -> None:
    if isinstance(instruction, ConstOp):
        return
    if isinstance(instruction, UnaryOp):
        instruction.value = _rewrite_operand(instruction.value, stacks)
        return
    if isinstance(instruction, BinaryOp):
        instruction.lhs = _rewrite_operand(instruction.lhs, stacks)
        instruction.rhs = _rewrite_operand(instruction.rhs, stacks)
        return
    if isinstance(instruction, CompareOp):
        instruction.lhs = _rewrite_operand(instruction.lhs, stacks)
        instruction.rhs = _rewrite_operand(instruction.rhs, stacks)
        return
    if isinstance(instruction, LoadOp):
        instruction.index = _rewrite_operand(instruction.index, stacks)
        return
    if isinstance(instruction, StoreOp):
        instruction.index = _rewrite_operand(instruction.index, stacks)
        instruction.value = _rewrite_operand(instruction.value, stacks)
        return
    if isinstance(instruction, PhiOp):
        return
    if isinstance(instruction, CallOp):
        instruction.operands = [_rewrite_operand(operand, stacks) for operand in instruction.operands]
        return
    if isinstance(instruction, PrintOp):
        instruction.operands = [_rewrite_operand(operand, stacks) for operand in instruction.operands]
        return


def _rename_terminator(terminator: object, stacks: dict[str, list[str]]) -> None:
    if isinstance(terminator, BranchOp):
        return
    if isinstance(terminator, CondBranchOp):
        terminator.cond = _rewrite_operand(terminator.cond, stacks)
        return
    if isinstance(terminator, ReturnOp) and terminator.value is not None:
        terminator.value = _rewrite_operand(terminator.value, stacks)


def _rewrite_operand(operand: object, stacks: dict[str, list[str]]) -> object:
    if isinstance(operand, Literal):
        return replace(operand)
    if isinstance(operand, Variable):
        name = operand.name
        if name not in stacks:
            return replace(operand)
        return Variable(_current_name(name, stacks), operand.type)
    if isinstance(operand, Parameter):
        if isinstance(normalize_type(operand.type), ArrayType) or operand.name not in stacks:
            return replace(operand)
        return Variable(_current_name(operand.name, stacks), operand.type)
    if isinstance(operand, str):
        if operand not in stacks:
            return operand
        return _current_name(operand, stacks)
    return deepcopy(operand)


def _new_name(base: str, counters: dict[str, int], stacks: dict[str, list[str]]) -> str:
    version = counters[base]
    counters[base] += 1
    name = f"{base}_{version}"
    stacks[base].append(name)
    return name


def _current_name(base: str, stacks: dict[str, list[str]]) -> str:
    stack = stacks.get(base)
    if not stack:
        raise UIRConstructionError(f"use of scalar '{base}' before any dominating definition")
    return stack[-1]


def _base_name(name: str) -> str:
    return name
