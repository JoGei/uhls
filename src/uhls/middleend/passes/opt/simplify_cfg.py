"""CFG simplification helpers for explicit CFG IR."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from uhls.middleend.uir import Block, BranchOp, CondBranchOp, Function, Module
from uhls.middleend.passes.analyze import control_flow
from uhls.middleend.passes.util.pass_manager import function_or_module_pass


def simplify_cfg_function(function: Function) -> Function:
    """Return a cleaned-up function with deterministic reachable CFG order."""
    result = deepcopy(function)

    changed = True
    while changed:
        changed = False
        reachable = _reachable_labels(result)
        if len(reachable) != len(result.blocks):
            result.blocks = [block for block in result.blocks if block.label in reachable]
            changed = True

        cfg = control_flow(result)
        raw_trampoline_map = {
            block.label: block.terminator.target
            for block in result.blocks
            if block.label != result.entry
            and not block.instructions
            and isinstance(block.terminator, BranchOp)
            and block.terminator.target != block.label
        }
        trampoline_map = {
            label: target
            for label, target in raw_trampoline_map.items()
            if target not in raw_trampoline_map
            and _can_bypass_trampoline(result, cfg, trampoline_label=label, target_label=target)
        }
        if trampoline_map:
            for block in result.blocks:
                terminator = block.terminator
                if isinstance(terminator, BranchOp):
                    terminator.target = _follow_target(terminator.target, trampoline_map)
                elif isinstance(terminator, CondBranchOp):
                    terminator.true_target = _follow_target(terminator.true_target, trampoline_map)
                    terminator.false_target = _follow_target(terminator.false_target, trampoline_map)
                    if terminator.true_target == terminator.false_target:
                        block.terminator = BranchOp(terminator.true_target)
            _rewrite_phi_incoming_after_trampoline_bypass(result, cfg, trampoline_map)
            changed = True
            continue

        predecessors = cfg.predecessors
        merged = False
        for block in result.blocks:
            if not isinstance(block.terminator, BranchOp):
                continue
            successor = _block_map(result).get(block.terminator.target)
            if successor is None or successor.label == result.entry:
                continue
            if successor.instructions and getattr(successor.instructions[0], "opcode", None) == "phi":
                continue
            if len(predecessors.get(successor.label, set())) != 1:
                continue
            if not isinstance(successor.terminator, (BranchOp, CondBranchOp)) and successor.terminator is None:
                continue

            block.instructions.extend(deepcopy(successor.instructions))
            block.terminator = deepcopy(successor.terminator)
            _rewrite_phi_predecessor_labels(result, old_label=successor.label, new_label=block.label)
            result.blocks = [candidate for candidate in result.blocks if candidate.label != successor.label]
            changed = True
            merged = True
            break

        if merged:
            continue

    reachable_order = _reachable_order(result)
    block_map = _block_map(result)
    result.blocks = [block_map[label] for label in reachable_order]
    return result


def simplify_cfg_module(module: Module) -> Module:
    """Simplify every function CFG in a module."""
    result = deepcopy(module)
    result.functions = [simplify_cfg_function(function) for function in result.functions]
    return result


def SimplifyCFGPass():
    """Return a pass-manager compatible CFG simplification pass."""
    return function_or_module_pass("simplify_cfg", simplify_cfg_function, simplify_cfg_module)


def _block_map(function: Function) -> dict[str, Block]:
    return {block.label: block for block in function.blocks}


def _reachable_labels(function: Function) -> set[str]:
    block_map = _block_map(function)
    worklist = [function.entry]
    seen: set[str] = set()
    while worklist:
        label = worklist.pop()
        if label in seen or label not in block_map:
            continue
        seen.add(label)
        terminator = block_map[label].terminator
        if isinstance(terminator, BranchOp):
            worklist.append(terminator.target)
        elif isinstance(terminator, CondBranchOp):
            worklist.append(terminator.false_target)
            worklist.append(terminator.true_target)
    return seen


def _reachable_order(function: Function) -> list[str]:
    block_map = _block_map(function)
    original_index = {block.label: index for index, block in enumerate(function.blocks)}
    order: list[str] = []
    seen: set[str] = set()

    def visit(label: str) -> None:
        if label in seen or label not in block_map:
            return
        seen.add(label)
        order.append(label)
        terminator = block_map[label].terminator
        successors: list[str] = []
        if isinstance(terminator, BranchOp):
            successors = [terminator.target]
        elif isinstance(terminator, CondBranchOp):
            successors = [terminator.true_target, terminator.false_target]
        for target in sorted(successors, key=lambda item: original_index.get(item, 10**9)):
            visit(target)

    visit(function.entry)
    return order


def _follow_target(target: str, trampoline_map: dict[str, str]) -> str:
    seen: set[str] = set()
    current = target
    while current in trampoline_map and current not in seen:
        seen.add(current)
        current = trampoline_map[current]
    return current


def _can_bypass_trampoline(
    function: Function,
    cfg: object,
    *,
    trampoline_label: str,
    target_label: str,
) -> bool:
    target = _block_map(function).get(target_label)
    if target is None:
        return False
    trampoline_predecessors = cfg.predecessors.get(trampoline_label, set())
    if not trampoline_predecessors:
        return True

    for instruction in target.instructions:
        if getattr(instruction, "opcode", None) != "phi":
            break
        trampoline_items = [incoming for incoming in instruction.incoming if getattr(incoming, "pred", None) == trampoline_label]
        if len(trampoline_items) != 1:
            return False
        trampoline_value = trampoline_items[0].value
        for predecessor in trampoline_predecessors:
            for incoming in instruction.incoming:
                if getattr(incoming, "pred", None) == predecessor and incoming.value != trampoline_value:
                    return False
    return True


def _rewrite_phi_incoming_after_trampoline_bypass(
    function: Function,
    cfg: object,
    trampoline_map: dict[str, str],
) -> None:
    for block in function.blocks:
        for instruction in block.instructions:
            if getattr(instruction, "opcode", None) != "phi":
                break
            expanded = []
            for incoming in instruction.incoming:
                pred = getattr(incoming, "pred", None)
                if pred in trampoline_map and trampoline_map[pred] == block.label:
                    for predecessor in sorted(cfg.predecessors.get(pred, set())):
                        expanded.append(replace(incoming, pred=predecessor))
                else:
                    expanded.append(incoming)

            deduped = []
            seen_by_pred: dict[str, object] = {}
            for incoming in expanded:
                pred = getattr(incoming, "pred", None)
                if pred not in seen_by_pred:
                    seen_by_pred[pred] = incoming.value
                    deduped.append(incoming)
                    continue
                if seen_by_pred[pred] != incoming.value:
                    raise ValueError(
                        f"simplify_cfg cannot merge phi predecessor '{pred}' in block '{block.label}' with conflicting values"
                    )
            instruction.incoming = deduped


def _rewrite_phi_predecessor_labels(function: Function, *, old_label: str, new_label: str) -> None:
    for block in function.blocks:
        for instruction in block.instructions:
            if getattr(instruction, "opcode", None) != "phi":
                break
            instruction.incoming = [
                replace(incoming, pred=new_label) if getattr(incoming, "pred", None) == old_label else incoming
                for incoming in getattr(instruction, "incoming", [])
            ]
