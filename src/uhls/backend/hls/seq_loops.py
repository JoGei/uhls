"""Loop-specific helpers for branch-based seq-stage loop lowering."""

from __future__ import annotations

from dataclasses import dataclass

from uhls.middleend.passes.analyze import detect_loops
from uhls.middleend.uir import Function


@dataclass
class LoopSummary:
    """One top-level loop summary during seq lowering."""

    header: str
    body_region_id: str
    empty_region_id: str
    body: frozenset[str]


def detect_top_level_loops(
    function: Function,
) -> list[LoopSummary]:
    """Return top-level loops selected from the frontend loop analysis."""
    summaries: list[LoopSummary] = []
    loop_infos = detect_loops(function)
    for info in loop_infos:
        if any(info.body < other.body for other in loop_infos):
            continue
        summaries.append(
            LoopSummary(
                header=info.header,
                body_region_id=_loop_body_region_id(function.name, info.header),
                empty_region_id=_loop_empty_region_id(function.name, info.header),
                body=frozenset(info.body),
            )
        )
    return summaries


def lower_loop_body_unit(
    function: Function,
    loop: LoopSummary,
    *,
    parent_id: str,
    kind: str,
    make_unit,
    nop_node,
    append_instructions,
    connect_produced_nodes_to_sink,
):
    """Lower one explicit loop body SGU."""
    unit = make_unit(id=loop.body_region_id, kind=kind, parent=parent_id)
    source = nop_node("source")
    sink = nop_node("sink")
    unit.nodes.extend([source, sink])
    block_map = function.block_map()
    ordered_blocks = [
        block_map[block.label]
        for block in function.blocks
        if block.label in loop.body and block.label != loop.header
    ]
    node_defs: dict[str, str] = {}
    last_memory: dict[str, str] = {}
    produced_nodes: list[str] = []

    for block in ordered_blocks:
        _, block_nodes = append_instructions(
            unit,
            block.instructions,
            cursor=source.id,
            node_defs=node_defs,
            last_memory=last_memory,
        )
        produced_nodes.extend(block_nodes)

    connect_produced_nodes_to_sink(unit, produced_nodes)
    return unit


def lower_loop_empty_unit(
    loop: LoopSummary,
    *,
    parent_id: str,
    make_unit,
    nop_node,
    make_edge,
):
    """Lower one explicit loop-empty/exit SGU."""
    unit = make_unit(id=loop.empty_region_id, kind="empty", parent=parent_id)
    source = nop_node("source")
    sink = nop_node("sink")
    unit.nodes.extend([source, sink])
    unit.edges.append(make_edge("data", source.id, sink.id))
    return unit


def loop_defined_names(function: Function, loop: LoopSummary, *, instruction_dest) -> set[str]:
    """Return SSA names defined inside one loop body."""
    block_map = function.block_map()
    names: set[str] = set()
    for label in loop.body:
        block = block_map.get(label)
        if block is None:
            continue
        for instruction in getattr(block, "instructions", []):
            dest = instruction_dest(instruction)
            if dest is not None:
                names.add(dest)
    return names


def _loop_body_region_id(function_name: str, header: str) -> str:
    return f"loop_body_{_sanitize(function_name)}_{_sanitize(header)}"


def _loop_empty_region_id(function_name: str, header: str) -> str:
    return f"loop_empty_{_sanitize(function_name)}_{_sanitize(header)}"


def _sanitize(text: str) -> str:
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in text)
    if not safe:
        return "anon"
    if safe[0].isdigit():
        return f"_{safe}"
    return safe
