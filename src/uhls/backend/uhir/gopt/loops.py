"""Shared explicit-loop helpers for µhIR graph-optimizer passes."""

from __future__ import annotations

from dataclasses import dataclass

from uhls.backend.uhir.model import UHIREdge, UHIRDesign, UHIRNode, UHIRRegion


@dataclass(frozen=True)
class ExplicitLoopInfo:
    """One already-structuralized loop hierarchy inside seq-stage µhIR."""

    loop_id: str
    parent_region: UHIRRegion
    loop_node: UHIRNode
    header_region: UHIRRegion
    header_branch: UHIRNode | None
    body_region: UHIRRegion | None
    empty_region: UHIRRegion | None
    entry_edge: UHIREdge | None
    return_edge: UHIREdge | None
    body_edge: UHIREdge | None
    exit_edge: UHIREdge | None
    backedge: UHIREdge | None


def collect_explicit_loops(design: UHIRDesign) -> list[ExplicitLoopInfo]:
    """Return already explicit loop hierarchy from seq-stage µhIR."""
    region_by_id = {region.id: region for region in design.regions}
    explicit: list[ExplicitLoopInfo] = []
    next_loop_id = 0

    for parent_region in sorted(design.regions, key=lambda region: region.id):
        for loop_node in sorted(
            (node for node in parent_region.nodes if node.opcode == "loop"),
            key=lambda node: node.id,
        ):
            child_id = loop_node.attributes.get("child")
            if not isinstance(child_id, str):
                continue
            header_region = region_by_id.get(child_id)
            if header_region is None or header_region.kind != "loop":
                continue

            header_branch = next((node for node in header_region.nodes if node.opcode == "branch"), None)
            body_region = None
            empty_region = None
            body_edge = None
            exit_edge = None
            backedge = None

            if header_branch is not None:
                body_id = header_branch.attributes.get("true_child")
                empty_id = header_branch.attributes.get("false_child")
                if isinstance(body_id, str):
                    candidate = region_by_id.get(body_id)
                    if candidate is not None:
                        body_region = candidate
                if isinstance(empty_id, str):
                    candidate = region_by_id.get(empty_id)
                    if candidate is not None:
                        empty_region = candidate
                body_edge = _find_edge(header_region, header_branch.id, body_id, when=True)
                exit_edge = _find_edge(header_region, header_branch.id, empty_id, when=False)
                if body_region is not None:
                    backedge = _find_edge(header_region, body_region.id, header_branch.id, when=True)

            explicit.append(
                ExplicitLoopInfo(
                    loop_id=f"L{next_loop_id}",
                    parent_region=parent_region,
                    loop_node=loop_node,
                    header_region=header_region,
                    header_branch=header_branch,
                    body_region=body_region,
                    empty_region=empty_region,
                    entry_edge=_find_edge(parent_region, loop_node.id, header_region.id),
                    return_edge=_find_edge(parent_region, header_region.id, loop_node.id),
                    body_edge=body_edge,
                    exit_edge=exit_edge,
                    backedge=backedge,
                )
            )
            next_loop_id += 1

    return explicit


def _find_edge(
    region: UHIRRegion,
    source: str,
    target: str | None,
    *,
    when: bool | None = None,
) -> UHIREdge | None:
    if target is None:
        return None
    for edge in region.edges:
        if edge.source != source or edge.target != target:
            continue
        if when is not None and edge.attributes.get("when") != when:
            continue
        return edge
    return None
