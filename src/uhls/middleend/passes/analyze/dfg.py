"""DFG-oriented analyses for canonical µhLS IR."""

from __future__ import annotations

from dataclasses import dataclass

from uhls.middleend.uir import (
    BinaryOp,
    CallOp,
    CompareOp,
    CondBranchOp,
    Function,
    Literal,
    LoadOp,
    Parameter,
    PhiOp,
    PrintOp,
    ReturnOp,
    StoreOp,
    UnaryOp,
    Variable,
)
from uhls.middleend.passes.util.pass_manager import AnalysisPass, analysis_pass


@dataclass(frozen=True)
class DFGEdge:
    """One dependency edge in a basic-block DFG."""

    source: str
    target: str
    kind: str
    label: str


@dataclass(frozen=True)
class DFGNode:
    """One instruction or terminator node in a basic-block DFG."""

    id: str
    block: str
    index: int
    opcode: str
    defines: tuple[str, ...]
    uses: tuple[str, ...]


@dataclass(frozen=True)
class BasicBlockDFG:
    """A block-local DFG with producer/consumer adjacency."""

    function: Function
    block: str
    nodes: tuple[DFGNode, ...]
    edges: tuple[DFGEdge, ...]
    predecessors: dict[str, set[str]]
    successors: dict[str, set[str]]


@dataclass(frozen=True)
class DFGInfo:
    """Block-local DFGs for one function."""

    function: Function
    blocks: dict[str, BasicBlockDFG]


def dfg(function: Function) -> DFGInfo:
    """Build block-local DFGs for one function."""
    return build_dfg(function)


def build_dfg(function: Function) -> DFGInfo:
    """Build one conservative basic-block DFG per function block."""
    return DFGInfo(
        function=function,
        blocks={block.label: build_block_dfg(function, block.label) for block in function.blocks},
    )


def build_block_dfg(function: Function, block_label: str) -> BasicBlockDFG:
    """Build a DFG for one basic block in ``function``."""
    block = function.block_map()[block_label]
    nodes: list[DFGNode] = []
    edges: list[DFGEdge] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    last_scalar_def: dict[str, str] = {}
    # Conservatively serialize repeated accesses to the same named array.
    last_memory_access: dict[str, str] = {}

    for index, instruction in enumerate(block.instructions):
        node_id = f"{block.label}:{index}"
        node = DFGNode(
            id=node_id,
            block=block.label,
            index=index,
            opcode=_opcode(instruction),
            defines=_instruction_defines(instruction),
            uses=_instruction_uses(instruction),
        )
        nodes.append(node)
        _connect_value_dependencies(edges, seen_edges, node_id, node.uses, last_scalar_def)
        _connect_memory_dependencies(
            edges,
            seen_edges,
            node_id,
            _instruction_memory_arrays(instruction),
            last_memory_access,
        )
        for name in node.defines:
            last_scalar_def[name] = node_id

    terminator_id = f"{block.label}:term"
    terminator_node = DFGNode(
        id=terminator_id,
        block=block.label,
        index=len(block.instructions),
        opcode=_opcode(block.terminator),
        defines=(),
        uses=_terminator_uses(block.terminator),
    )
    nodes.append(terminator_node)
    _connect_value_dependencies(edges, seen_edges, terminator_id, terminator_node.uses, last_scalar_def)
    _connect_terminator_convergence(edges, seen_edges, nodes, terminator_id)

    predecessors = {node.id: set() for node in nodes}
    successors = {node.id: set() for node in nodes}
    for edge in edges:
        successors[edge.source].add(edge.target)
        predecessors[edge.target].add(edge.source)

    return BasicBlockDFG(
        function=function,
        block=block.label,
        nodes=tuple(nodes),
        edges=tuple(edges),
        predecessors=predecessors,
        successors=successors,
    )


def dfg_pass(key: str = "dfg") -> AnalysisPass:
    """Return a reusable DFG analysis pass."""
    return analysis_pass("dfg", build_dfg, key=key)


def _connect_value_dependencies(
    edges: list[DFGEdge],
    seen_edges: set[tuple[str, str, str, str]],
    target: str,
    uses: tuple[str, ...],
    last_scalar_def: dict[str, str],
) -> None:
    for name in uses:
        source = last_scalar_def.get(name)
        if source is not None:
            _append_edge(edges, seen_edges, source, target, "value", name)


def _connect_memory_dependencies(
    edges: list[DFGEdge],
    seen_edges: set[tuple[str, str, str, str]],
    target: str,
    arrays: tuple[str, ...],
    last_memory_access: dict[str, str],
) -> None:
    for array in arrays:
        source = last_memory_access.get(array)
        if source is not None:
            _append_edge(edges, seen_edges, source, target, "memory", array)
        last_memory_access[array] = target


def _append_edge(
    edges: list[DFGEdge],
    seen_edges: set[tuple[str, str, str, str]],
    source: str,
    target: str,
    kind: str,
    label: str,
) -> None:
    key = (source, target, kind, label)
    if key in seen_edges:
        return
    seen_edges.add(key)
    edges.append(DFGEdge(source=source, target=target, kind=kind, label=label))


def _connect_terminator_convergence(
    edges: list[DFGEdge],
    seen_edges: set[tuple[str, str, str, str]],
    nodes: list[DFGNode],
    terminator_id: str,
) -> None:
    non_terminator_ids = {node.id for node in nodes if node.id != terminator_id}
    internal_sources = {edge.source for edge in edges if edge.target != terminator_id}
    direct_terminator_sources = {edge.source for edge in edges if edge.target == terminator_id and edge.kind != "sink"}
    leaf_ids = sorted(non_terminator_ids - internal_sources)
    for node_id in leaf_ids:
        if node_id in direct_terminator_sources:
            continue
        _append_edge(edges, seen_edges, node_id, terminator_id, "sink", "")


def _instruction_defines(instruction: object) -> tuple[str, ...]:
    dest = getattr(instruction, "dest", None)
    if isinstance(dest, str):
        return (dest,)
    return ()


def _instruction_uses(instruction: object) -> tuple[str, ...]:
    if isinstance(instruction, PhiOp):
        return ()
    if isinstance(instruction, UnaryOp):
        return _operand_names(instruction.value)
    if isinstance(instruction, (BinaryOp, CompareOp)):
        return _operand_names(instruction.lhs) + _operand_names(instruction.rhs)
    if isinstance(instruction, LoadOp):
        return _operand_names(instruction.index)
    if isinstance(instruction, StoreOp):
        return _operand_names(instruction.index) + _operand_names(instruction.value)
    if isinstance(instruction, CallOp):
        names: list[str] = []
        for operand in instruction.operands:
            names.extend(_operand_names(operand))
        return tuple(names)
    if isinstance(instruction, PrintOp):
        names: list[str] = []
        for operand in instruction.operands:
            names.extend(_operand_names(operand))
        return tuple(names)
    return ()


def _instruction_memory_arrays(instruction: object) -> tuple[str, ...]:
    if isinstance(instruction, (LoadOp, StoreOp)):
        return (instruction.array,)
    return ()


def _terminator_uses(terminator: object) -> tuple[str, ...]:
    if isinstance(terminator, CondBranchOp):
        return _operand_names(terminator.cond)
    if isinstance(terminator, ReturnOp) and terminator.value is not None:
        return _operand_names(terminator.value)
    return ()


def _operand_names(operand: object) -> tuple[str, ...]:
    if isinstance(operand, Literal):
        return ()
    if isinstance(operand, Variable):
        return (operand.name,)
    if isinstance(operand, Parameter):
        return (operand.name,)
    if isinstance(operand, str):
        return (operand,)
    return ()


def _opcode(operation: object) -> str:
    return str(getattr(operation, "opcode", operation.__class__.__name__))
