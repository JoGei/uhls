"""Loop canonicalization helpers for canonical µhLS IR."""

from __future__ import annotations

from copy import deepcopy

from uhls.middleend.passes.analyze import control_flow, detect_loops
from uhls.middleend.passes.util.pass_manager import function_or_module_pass
from uhls.middleend.uir import Block, BranchOp, CondBranchOp, Function, IncomingValue, Module, PhiOp


def canonicalize_loops_function(function: Function) -> Function:
    """Rewrite loops into a simpler canonical shape with one latch per header."""
    result = deepcopy(function)

    changed = True
    while changed:
        changed = False
        cfg = control_flow(result)
        for loop in detect_loops(result):
            in_loop_predecessors = tuple(sorted(cfg.predecessors[loop.header] & set(loop.body)))
            if len(in_loop_predecessors) <= 1:
                continue
            _canonicalize_multi_latch_loop(result, loop.header, in_loop_predecessors)
            changed = True
            break

    return result


def canonicalize_loops_module(module: Module) -> Module:
    """Canonicalize loops across every function in one module."""
    result = deepcopy(module)
    result.functions = [canonicalize_loops_function(function) for function in result.functions]
    return result


def CanonicalizeLoopsPass():
    """Return a pass-manager compatible loop canonicalization pass."""
    return function_or_module_pass("canonicalize_loops", canonicalize_loops_function, canonicalize_loops_module)


def _canonicalize_multi_latch_loop(function: Function, header_label: str, in_loop_predecessors: tuple[str, ...]) -> None:
    block_map = function.block_map()
    header = block_map[header_label]
    latch_label = _fresh_label(f"{header_label}_latch", {block.label for block in function.blocks})
    used_symbols = _collect_used_symbols(function)

    latch_instructions: list[object] = []
    for instruction in header.instructions:
        if not isinstance(instruction, PhiOp):
            break
        latch_value_name = _fresh_symbol(f"{instruction.dest}_latch", used_symbols)
        latch_incoming = [
            IncomingValue(incoming.pred, incoming.value)
            for incoming in instruction.incoming
            if incoming.pred in in_loop_predecessors
        ]
        latch_instructions.append(PhiOp(latch_value_name, instruction.type, latch_incoming))
        instruction.incoming = [
            incoming for incoming in instruction.incoming if incoming.pred not in in_loop_predecessors
        ] + [IncomingValue(latch_label, latch_value_name)]

    for predecessor_label in in_loop_predecessors:
        predecessor = block_map[predecessor_label]
        terminator = predecessor.terminator
        if isinstance(terminator, BranchOp):
            if terminator.target == header_label:
                terminator.target = latch_label
            continue
        if isinstance(terminator, CondBranchOp):
            if terminator.true_target == header_label:
                terminator.true_target = latch_label
            if terminator.false_target == header_label:
                terminator.false_target = latch_label

    insertion_index = 1 + max(index for index, block in enumerate(function.blocks) if block.label in in_loop_predecessors)
    function.blocks.insert(insertion_index, Block(latch_label, latch_instructions, BranchOp(header_label)))


def _collect_used_symbols(function: Function) -> set[str]:
    used = {parameter.name for parameter in function.params}
    used.update(function.local_arrays)
    for block in function.blocks:
        for instruction in block.instructions:
            dest = getattr(instruction, "dest", None)
            if isinstance(dest, str):
                used.add(dest)
    return used


def _fresh_symbol(base: str, used: set[str]) -> str:
    candidate = base
    index = 1
    while candidate in used:
        index += 1
        candidate = f"{base}_{index}"
    used.add(candidate)
    return candidate


def _fresh_label(base: str, used: set[str]) -> str:
    candidate = base
    index = 1
    while candidate in used:
        index += 1
        candidate = f"{base}_{index}"
    used.add(candidate)
    return candidate
