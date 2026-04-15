"""Loop unrolling for canonical single-block counted loops."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from uhls.middleend.passes.analyze import control_flow, detect_loops
from uhls.middleend.passes.util.pass_manager import PassContext
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
    LoadOp,
    Module,
    ParamOp,
    PhiOp,
    PrintOp,
    StoreOp,
    UnaryOp,
    Variable,
)


def unroll_loops_function(function: Function, target_loop: str, count: int) -> Function:
    """Unroll one canonical single-block loop identified by its header name."""
    if count <= 1:
        raise ValueError(f"unroll count must be greater than 1, got {count}")

    result = deepcopy(function)
    target_header = _resolve_target_header(result, target_loop)
    cfg = control_flow(result)
    loops = {loop.header: loop for loop in detect_loops(result)}
    loop = loops.get(target_header)
    if loop is None:
        raise ValueError(f"target loop '{target_loop}' does not identify a natural loop in function '{function.name}'")
    if len(loop.latches) != 1:
        raise ValueError(f"loop '{target_header}' is not supported: expected exactly one latch block")

    block_map = result.block_map()
    header = block_map[target_header]
    latch_label = loop.latches[0]
    if loop.body != frozenset({target_header, latch_label}):
        raise ValueError(f"loop '{target_header}' is not supported: expected one header block and one body block")

    if not isinstance(header.terminator, CondBranchOp):
        raise ValueError(f"loop '{target_header}' is not supported: header terminator must be conditional")
    latch = block_map[latch_label]
    if not isinstance(latch.terminator, BranchOp) or latch.terminator.target != target_header:
        raise ValueError(f"loop '{target_header}' is not supported: latch must branch directly back to the header")

    loop_successors = cfg.successors[target_header] & set(loop.body)
    if loop_successors != {latch_label}:
        raise ValueError(f"loop '{target_header}' is not supported: header must branch to exactly one in-loop body")

    preheader_candidates = cfg.predecessors[target_header] - set(loop.body)
    if len(preheader_candidates) != 1:
        raise ValueError(f"loop '{target_header}' is not supported: expected exactly one loop preheader")

    phi_instructions = _leading_phi_instructions(header)
    if not phi_instructions:
        raise ValueError(f"loop '{target_header}' is not supported: expected loop-carried phi nodes in the header")

    header_tail = list(header.instructions[len(phi_instructions) :])
    original_body_instructions = list(latch.instructions)
    continue_on_true = header.terminator.true_target == latch_label

    loop_carried_values: dict[str, object] = {}
    for phi in phi_instructions:
        incoming = {item.pred: item.value for item in phi.incoming}
        if latch_label not in incoming:
            raise ValueError(f"loop '{target_header}' is not supported: phi '{phi.dest}' is missing latch input")
        loop_carried_values[phi.dest] = incoming[latch_label]

    used_labels = {block.label for block in result.blocks}
    used_symbols = _collect_used_symbols(result)

    current_state = dict(loop_carried_values)
    first_guard_mapping = dict(current_state)
    first_guard_suffix = _unroll_suffix(1)
    cloned_header_instructions = [
        _clone_instruction_once(instruction, first_guard_mapping, used_symbols, first_guard_suffix)
        for instruction in header_tail
    ]
    cloned_cond = _remap_operand(header.terminator.cond, first_guard_mapping)

    first_stage_label = _fresh_name(f"{latch_label}_unroll_1", used_labels)
    latch.instructions.extend(cloned_header_instructions)
    latch.terminator = (
        CondBranchOp(cloned_cond, first_stage_label, header.label)
        if continue_on_true
        else CondBranchOp(cloned_cond, header.label, first_stage_label)
    )

    new_blocks: list[Block] = []
    next_stage_label = first_stage_label
    for stage_index in range(1, count):
        stage_mapping = dict(current_state)
        stage_suffix = _unroll_suffix(stage_index)
        cloned_body_instructions = [
            _clone_instruction_once(instruction, stage_mapping, used_symbols, stage_suffix)
            for instruction in original_body_instructions
        ]

        for phi in phi_instructions:
            phi.incoming.append(
                IncomingValue(next_stage_label, _remap_operand(loop_carried_values[phi.dest], stage_mapping))
            )

        if stage_index < count - 1:
            current_state = {
                phi.dest: _remap_operand(loop_carried_values[phi.dest], stage_mapping)
                for phi in phi_instructions
            }
            guard_mapping = dict(current_state)
            guard_suffix = _unroll_suffix(stage_index + 1)
            cloned_guard_instructions = [
                _clone_instruction_once(instruction, guard_mapping, used_symbols, guard_suffix)
                for instruction in header_tail
            ]
            cloned_body_instructions.extend(cloned_guard_instructions)
            cloned_stage_cond = _remap_operand(header.terminator.cond, guard_mapping)
            following_stage_label = _fresh_name(f"{latch_label}_unroll_{stage_index + 1}", used_labels)
            terminator = (
                CondBranchOp(cloned_stage_cond, following_stage_label, header.label)
                if continue_on_true
                else CondBranchOp(cloned_stage_cond, header.label, following_stage_label)
            )
            new_blocks.append(Block(next_stage_label, cloned_body_instructions, terminator))
            next_stage_label = following_stage_label
            continue

        new_blocks.append(Block(next_stage_label, cloned_body_instructions, BranchOp(header.label)))

    body_index = next(index for index, block in enumerate(result.blocks) if block.label == latch_label)
    result.blocks[body_index + 1 : body_index + 1] = new_blocks
    return result


def unroll_loops_module(module: Module, target_loop: str, count: int) -> Module:
    """Unroll one uniquely identified loop within one module."""
    result = deepcopy(module)
    matching_indices = [
        index
        for index, function in enumerate(result.functions)
        if _maybe_resolve_target_header(function, target_loop) is not None
    ]
    if not matching_indices:
        raise ValueError(f"target loop '{target_loop}' was not found in the module")
    if len(matching_indices) > 1:
        raise ValueError(f"target loop '{target_loop}' matched multiple functions; apply the pass to one function instead")
    index = matching_indices[0]
    result.functions[index] = unroll_loops_function(result.functions[index], target_loop, count)
    return result


@dataclass(frozen=True)
class _UnrollLoopsPass:
    target_loop: str | None = None
    count: int | None = None
    name: str = "unroll_loops"

    def run(self, ir: Function | Module, context: PassContext, pass_args: tuple[str, ...] = ()) -> Function | Module:
        target_loop, count = _resolve_config(self.target_loop, self.count, pass_args)
        if hasattr(ir, "functions"):
            return unroll_loops_module(ir, target_loop, count)
        return unroll_loops_function(ir, target_loop, count)


def UnrollLoopsPass(target_loop: str | None = None, count: int | None = None) -> _UnrollLoopsPass:
    """Return a pass-manager compatible loop-unroll pass."""
    return _UnrollLoopsPass(target_loop=target_loop, count=count)


def _resolve_config(
    target_loop: str | None,
    count: int | None,
    pass_args: tuple[str, ...],
) -> tuple[str, int]:
    active_target = target_loop
    active_count = count
    if active_target is None or active_count is None:
        if len(pass_args) != 2:
            raise ValueError("unroll_loops expects exactly two arguments: <target-loop> <count>")
        if active_target is None:
            active_target = pass_args[0]
        if active_count is None:
            active_count = int(pass_args[1])
    assert active_target is not None
    assert active_count is not None
    if active_count <= 1:
        raise ValueError(f"unroll count must be greater than 1, got {active_count}")
    return active_target, active_count


def _resolve_target_header(function: Function, target_loop: str) -> str:
    resolved = _maybe_resolve_target_header(function, target_loop)
    if resolved is None:
        raise ValueError(f"target loop '{target_loop}' was not found in function '{function.name}'")
    return resolved


def _maybe_resolve_target_header(function: Function, target_loop: str) -> str | None:
    loop_headers = {loop.header for loop in detect_loops(function)}
    if target_loop in loop_headers:
        return target_loop
    prefixed = f"for_header_{target_loop}"
    if prefixed in loop_headers:
        return prefixed
    return None


def _leading_phi_instructions(block: Block) -> list[PhiOp]:
    phis: list[PhiOp] = []
    for instruction in block.instructions:
        if not isinstance(instruction, PhiOp):
            break
        phis.append(instruction)
    return phis


def _collect_used_symbols(function: Function) -> set[str]:
    used = {parameter.name for parameter in function.params}
    used.update(function.local_arrays)
    for block in function.blocks:
        for instruction in block.instructions:
            dest = getattr(instruction, "dest", None)
            if isinstance(dest, str):
                used.add(dest)
    return used


def _fresh_name(base: str, used: set[str]) -> str:
    candidate = base
    index = 1
    while candidate in used:
        index += 1
        candidate = f"{base}_{index}"
    used.add(candidate)
    return candidate


def _unroll_suffix(stage_index: int) -> str:
    return f"__u{stage_index}"


def _remap_operand(operand: object, mapping: dict[str, object]) -> object:
    if isinstance(operand, Variable):
        return Variable(mapping.get(operand.name, operand.name), operand.type)
    if isinstance(operand, str):
        return mapping.get(operand, operand)
    return deepcopy(operand)


def _clone_instruction_once(
    instruction: object,
    mapping: dict[str, object],
    used_symbols: set[str],
    suffix: str,
) -> object:
    if isinstance(instruction, PhiOp):
        raise ValueError("unroll_loops only supports cloning non-phi instructions outside the loop header")
    if isinstance(instruction, ConstOp):
        dest = _fresh_name(f"{instruction.dest}{suffix}", used_symbols)
        mapping[instruction.dest] = dest
        return ConstOp(dest, instruction.type, instruction.value)
    if isinstance(instruction, UnaryOp):
        dest = _fresh_name(f"{instruction.dest}{suffix}", used_symbols)
        mapping[instruction.dest] = dest
        return UnaryOp(instruction.opcode, dest, instruction.type, _remap_operand(instruction.value, mapping))
    if isinstance(instruction, BinaryOp):
        dest = _fresh_name(f"{instruction.dest}{suffix}", used_symbols)
        mapping[instruction.dest] = dest
        return BinaryOp(
            instruction.opcode,
            dest,
            instruction.type,
            _remap_operand(instruction.lhs, mapping),
            _remap_operand(instruction.rhs, mapping),
        )
    if isinstance(instruction, CompareOp):
        dest = _fresh_name(f"{instruction.dest}{suffix}", used_symbols)
        mapping[instruction.dest] = dest
        return CompareOp(
            instruction.opcode,
            dest,
            _remap_operand(instruction.lhs, mapping),
            _remap_operand(instruction.rhs, mapping),
            type=instruction.type,
        )
    if isinstance(instruction, LoadOp):
        dest = _fresh_name(f"{instruction.dest}{suffix}", used_symbols)
        mapping[instruction.dest] = dest
        return LoadOp(dest, instruction.type, instruction.array, _remap_operand(instruction.index, mapping))
    if isinstance(instruction, StoreOp):
        return StoreOp(
            instruction.array,
            _remap_operand(instruction.index, mapping),
            _remap_operand(instruction.value, mapping),
        )
    if isinstance(instruction, CallOp):
        dest = instruction.dest
        if dest is not None:
            dest = _fresh_name(f"{dest}{suffix}", used_symbols)
            mapping[instruction.dest] = dest
        return CallOp(
            instruction.callee,
            [_remap_operand(operand, mapping) for operand in instruction.operands],
            dest=dest,
            type=instruction.type,
            arg_count=instruction.arg_count,
        )
    if isinstance(instruction, PrintOp):
        return PrintOp(instruction.format, [_remap_operand(operand, mapping) for operand in instruction.operands])
    if isinstance(instruction, ParamOp):
        return ParamOp(instruction.index, _remap_operand(instruction.value, mapping))
    raise TypeError(f"unsupported instruction for loop unrolling: {type(instruction).__name__}")
