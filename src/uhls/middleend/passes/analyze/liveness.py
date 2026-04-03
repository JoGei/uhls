"""Scalar liveness analyses for canonical µhLS IR."""

from __future__ import annotations

from dataclasses import dataclass

from uhls.middleend.uir import (
    ArrayType,
    BinaryOp,
    CallOp,
    CompareOp,
    CondBranchOp,
    Function,
    Literal,
    LoadOp,
    Parameter,
    PhiOp,
    ReturnOp,
    StoreOp,
    UnaryOp,
    Variable,
    normalize_type,
)
from uhls.middleend.passes.analyze.cfg import control_flow
from uhls.middleend.passes.util.pass_manager import AnalysisPass, analysis_pass


@dataclass(frozen=True)
class LivenessInfo:
    """Block-local scalar liveness information."""

    uses: dict[str, set[str]]
    defs: dict[str, set[str]]
    live_in: dict[str, set[str]]
    live_out: dict[str, set[str]]


def liveness(function: Function) -> LivenessInfo:
    """Compute phi-aware scalar liveness for one function."""
    cfg = control_flow(function)
    symbol_names = _scalar_symbol_names(function)
    uses = {block.label: set() for block in function.blocks}
    defs = {block.label: set() for block in function.blocks}
    phi_defs = {block.label: set() for block in function.blocks}
    phi_edge_uses = {
        block.label: {succ: set() for succ in cfg.successors[block.label]}
        for block in function.blocks
    }

    for block in function.blocks:
        for instruction in block.instructions:
            if isinstance(instruction, PhiOp):
                phi_defs[block.label].add(instruction.dest)
                defs[block.label].add(instruction.dest)
                for incoming in instruction.incoming:
                    if incoming.pred in phi_edge_uses:
                        phi_edge_uses[incoming.pred].setdefault(block.label, set()).update(
                            _operand_scalar_names(incoming.value, symbol_names)
                        )
                continue

            for name in _instruction_uses(instruction, symbol_names):
                if name not in defs[block.label]:
                    uses[block.label].add(name)
            dest = getattr(instruction, "dest", None)
            if isinstance(dest, str) and dest in symbol_names:
                defs[block.label].add(dest)

        for name in _terminator_uses(block.terminator, symbol_names):
            if name not in defs[block.label]:
                uses[block.label].add(name)

    live_in = {block.label: set() for block in function.blocks}
    live_out = {block.label: set() for block in function.blocks}

    changed = True
    while changed:
        changed = False
        for block in reversed(function.blocks):
            label = block.label
            new_live_out = set()
            for succ in cfg.successors[label]:
                edge_uses = phi_edge_uses.get(label, {}).get(succ, set())
                new_live_out |= (live_in[succ] - phi_defs[succ]) | edge_uses

            new_live_in = uses[label] | (new_live_out - defs[label])
            if new_live_in != live_in[label] or new_live_out != live_out[label]:
                live_in[label] = new_live_in
                live_out[label] = new_live_out
                changed = True

    return LivenessInfo(uses=uses, defs=defs, live_in=live_in, live_out=live_out)


def liveliness(function: Function) -> LivenessInfo:
    """Compatibility alias for callers using the 'liveliness' spelling."""
    return liveness(function)


def liveness_pass(key: str = "liveness") -> AnalysisPass:
    """Return a reusable liveness analysis pass."""
    return analysis_pass("liveness", liveness, key=key)


def liveliness_pass(key: str = "liveliness") -> AnalysisPass:
    """Return a reusable liveliness analysis pass."""
    return analysis_pass("liveliness", liveliness, key=key)


def _scalar_symbol_names(function: Function) -> set[str]:
    names: set[str] = set()
    for param in function.params:
        if not isinstance(normalize_type(param.type), ArrayType):
            names.add(param.name)
    for block in function.blocks:
        for instruction in block.instructions:
            dest = getattr(instruction, "dest", None)
            if isinstance(dest, str):
                names.add(dest)
    return names


def _instruction_uses(instruction: object, symbol_names: set[str]) -> set[str]:
    if isinstance(instruction, UnaryOp):
        return _operand_scalar_names(instruction.value, symbol_names)
    if isinstance(instruction, (BinaryOp, CompareOp)):
        return _operand_scalar_names(instruction.lhs, symbol_names) | _operand_scalar_names(
            instruction.rhs, symbol_names
        )
    if isinstance(instruction, LoadOp):
        return _operand_scalar_names(instruction.index, symbol_names)
    if isinstance(instruction, StoreOp):
        return _operand_scalar_names(instruction.index, symbol_names) | _operand_scalar_names(
            instruction.value, symbol_names
        )
    if isinstance(instruction, CallOp):
        used: set[str] = set()
        for operand in instruction.operands:
            used |= _operand_scalar_names(operand, symbol_names)
        return used
    return set()


def _terminator_uses(terminator: object, symbol_names: set[str]) -> set[str]:
    if isinstance(terminator, CondBranchOp):
        return _operand_scalar_names(terminator.cond, symbol_names)
    if isinstance(terminator, ReturnOp) and terminator.value is not None:
        return _operand_scalar_names(terminator.value, symbol_names)
    return set()


def _operand_scalar_names(operand: object, symbol_names: set[str]) -> set[str]:
    if isinstance(operand, Literal):
        return set()
    if isinstance(operand, Variable):
        return {operand.name} if operand.name in symbol_names else set()
    if isinstance(operand, Parameter):
        return {operand.name} if operand.name in symbol_names else set()
    if isinstance(operand, str):
        return {operand} if operand in symbol_names else set()
    return set()
