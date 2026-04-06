"""Shared loop-analysis helpers for µhIR graph-optimizer passes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from uhls.backend.uhir.model import UHIREdge, UHIRDesign, UHIRNode, UHIRRegion, UHIRRegionRef


@dataclass(frozen=True)
class LoopCandidate:
    """One branch-based loop candidate inside seq-stage µhIR."""

    loop_id: str
    parent_region: UHIRRegion
    branch_node: UHIRNode
    body_region: UHIRRegion
    empty_region: UHIRRegion
    header_node_ids: frozenset[str]


@dataclass(frozen=True)
class ExplicitLoopInfo:
    """One already explicit loop hierarchy inside seq-stage µhIR."""

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


def collect_loop_candidates(design: UHIRDesign) -> list[LoopCandidate]:
    """Return branch-based loop candidates from seq-stage µhIR."""
    region_by_id = {region.id: region for region in design.regions}
    candidates: list[LoopCandidate] = []
    next_loop_id = 0

    for parent_region in sorted(design.regions, key=lambda region: region.id):
        node_by_id = {node.id: node for node in parent_region.nodes}
        incoming: dict[str, set[str]] = {node.id: set() for node in parent_region.nodes}
        for edge in parent_region.edges:
            if edge.kind != "data":
                continue
            if edge.source in node_by_id and edge.target in node_by_id:
                incoming.setdefault(edge.target, set()).add(edge.source)

        for branch_node in sorted((node for node in parent_region.nodes if node.opcode == "branch"), key=lambda node: node.id):
            true_child = branch_node.attributes.get("true_child")
            false_child = branch_node.attributes.get("false_child")
            if not isinstance(true_child, str) or not isinstance(false_child, str):
                continue
            body_region = region_by_id.get(true_child)
            empty_region = region_by_id.get(false_child)
            if body_region is None or empty_region is None:
                continue
            if body_region.parent != parent_region.id or empty_region.parent != parent_region.id:
                continue
            if empty_region.kind != "empty":
                continue

            body_names = produced_names(body_region)
            header_node_ids = _header_backward_closure(parent_region, branch_node.id)
            header_node_ids.update(_loop_header_phi_ids(parent_region, body_names))
            if not header_node_ids:
                continue
            header_nodes = [node_by_id[node_id] for node_id in header_node_ids if node_id in node_by_id]
            if not any(node.opcode == "phi" and any(operand in body_names for operand in node.operands) for node in header_nodes):
                continue

            candidates.append(
                LoopCandidate(
                    loop_id=f"L{next_loop_id}",
                    parent_region=parent_region,
                    branch_node=branch_node,
                    body_region=body_region,
                    empty_region=empty_region,
                    header_node_ids=frozenset(header_node_ids),
                )
            )
            next_loop_id += 1

    return candidates


def collect_explicit_loops(design: UHIRDesign) -> list[ExplicitLoopInfo]:
    """Return already explicit loop hierarchy from seq-stage µhIR."""
    region_by_id = {region.id: region for region in design.regions}
    explicit: list[ExplicitLoopInfo] = []
    next_loop_id = 0

    for parent_region in sorted(design.regions, key=lambda region: region.id):
        for loop_node in sorted((node for node in parent_region.nodes if node.opcode == "loop"), key=lambda node: node.id):
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
                    body_region = region_by_id.get(body_id)
                if isinstance(empty_id, str):
                    empty_region = region_by_id.get(empty_id)
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


def explicit_loop_region_id(candidate: LoopCandidate) -> str:
    """Return the explicit loop-header region id for one candidate."""
    body_id = candidate.body_region.id
    if body_id.startswith("loop_body_"):
        return "loop_" + body_id[len("loop_body_") :]
    return f"loop_{candidate.branch_node.id}"


def explicit_loop_from_candidate(design: UHIRDesign, candidate: LoopCandidate) -> UHIRDesign:
    """Convert one branch-based loop candidate into explicit loop dialect."""
    result = deepcopy(design)
    region_by_id = {region.id: region for region in result.regions}
    parent_region = region_by_id[candidate.parent_region.id]
    body_region = region_by_id[candidate.body_region.id]
    empty_region = region_by_id[candidate.empty_region.id]
    branch_node = next(node for node in parent_region.nodes if node.id == candidate.branch_node.id)

    header_region_id = explicit_loop_region_id(candidate)
    header_region = UHIRRegion(id=header_region_id, kind="loop", parent=parent_region.id)
    body_region.kind = "body"
    body_region.parent = header_region.id
    empty_region.parent = header_region.id

    next_id = _fresh_node_counter(result)
    source_id = f"v{next_id}"
    next_id += 1
    sink_id = f"v{next_id}"
    next_id += 1
    header_region.nodes = [
        UHIRNode(source_id, "nop", attributes={"role": "source"}),
        UHIRNode(sink_id, "nop", attributes={"role": "sink"}),
    ]

    node_ids_to_move = set(candidate.header_node_ids)
    moved_nodes = [node for node in parent_region.nodes if node.id in node_ids_to_move]
    moved_mappings = [mapping for mapping in parent_region.mappings if mapping.node_id in node_ids_to_move]
    moved_edges = [edge for edge in parent_region.edges if _edge_is_loop_header_internal(edge, node_ids_to_move, body_region.id, empty_region.id)]

    header_branch_id = f"v{next_id}"
    branch_id_map = {branch_node.id: header_branch_id}

    for node in moved_nodes:
        cloned = deepcopy(node)
        cloned.id = branch_id_map.get(node.id, node.id)
        header_region.nodes.append(cloned)
    for edge in moved_edges:
        cloned = deepcopy(edge)
        cloned.source = branch_id_map.get(cloned.source, cloned.source)
        cloned.target = branch_id_map.get(cloned.target, cloned.target)
        header_region.edges.append(cloned)
    header_region.mappings.extend(moved_mappings)
    header_region.region_refs.extend([UHIRRegionRef(body_region.id), UHIRRegionRef(empty_region.id)])
    _close_region_boundaries(header_region)

    parent_region.nodes = [node for node in parent_region.nodes if node.id not in node_ids_to_move]
    parent_region.mappings = [mapping for mapping in parent_region.mappings if mapping.node_id not in node_ids_to_move]
    parent_region.edges = [
        edge
        for edge in parent_region.edges
        if not _edge_touches_removed_header(
            edge,
            branch_node.id,
            node_ids_to_move - {branch_node.id},
            body_region.id,
            empty_region.id,
        )
    ]

    loop_node = UHIRNode(branch_node.id, "loop", attributes={"child": header_region.id})
    parent_region.nodes.append(loop_node)
    parent_region.region_refs = [ref for ref in parent_region.region_refs if ref.target not in {body_region.id, empty_region.id}]
    parent_region.region_refs.append(UHIRRegionRef(header_region.id))

    external_predecessors = sorted(
        {
            edge.source
            for edge in candidate.parent_region.edges
            if edge.kind == "data" and edge.target in node_ids_to_move and edge.source not in node_ids_to_move
        }
    )
    for predecessor in external_predecessors:
        parent_region.edges.append(UHIREdge("data", predecessor, loop_node.id))
    parent_region.edges.append(UHIREdge("seq", loop_node.id, header_region.id, {"hierarchy": True}))
    parent_region.edges.append(UHIREdge("seq", header_region.id, loop_node.id, {"hierarchy": True}))
    _dedupe_region_edges(parent_region)

    result.regions.append(header_region)
    return result


def produced_names(region: UHIRRegion) -> set[str]:
    """Return all names produced by one region."""
    names = {node.id for node in region.nodes}
    for mapping in region.mappings:
        names.add(mapping.source_id)
    return names


def _header_backward_closure(region: UHIRRegion, branch_id: str) -> set[str]:
    node_by_id = {node.id: node for node in region.nodes}
    incoming: dict[str, set[str]] = {node.id: set() for node in region.nodes}
    for edge in region.edges:
        if edge.kind != "data":
            continue
        if edge.source in node_by_id and edge.target in node_by_id:
            incoming.setdefault(edge.target, set()).add(edge.source)

    closure: set[str] = set()
    worklist = [branch_id]
    while worklist:
        node_id = worklist.pop()
        if node_id in closure or node_id not in node_by_id:
            continue
        closure.add(node_id)
        for predecessor in incoming.get(node_id, ()):
            predecessor_node = node_by_id.get(predecessor)
            if predecessor_node is None:
                continue
            if predecessor_node.opcode == "nop" and predecessor_node.attributes.get("role") == "source":
                continue
            worklist.append(predecessor)
    return closure


def _loop_header_phi_ids(region: UHIRRegion, body_names: set[str]) -> set[str]:
    phi_ids: set[str] = set()
    for node in region.nodes:
        if node.opcode != "phi":
            continue
        if any(operand in body_names for operand in node.operands):
            phi_ids.add(node.id)
    return phi_ids


def _edge_is_loop_header_internal(
    edge: UHIREdge,
    header_node_ids: set[str],
    body_region_id: str,
    empty_region_id: str,
) -> bool:
    if edge.source in header_node_ids and edge.target in header_node_ids:
        return True
    if edge.kind == "seq" and edge.source in header_node_ids and edge.target in {body_region_id, empty_region_id}:
        return True
    if edge.kind == "seq" and edge.target in header_node_ids and edge.source in {body_region_id, empty_region_id}:
        return True
    return False


def _edge_touches_removed_header(
    edge: UHIREdge,
    branch_node_id: str,
    removed_node_ids: set[str],
    body_region_id: str,
    empty_region_id: str,
) -> bool:
    if edge.source in removed_node_ids or edge.target in removed_node_ids:
        return True
    if edge.kind == "seq" and edge.source == branch_node_id and edge.target in {body_region_id, empty_region_id}:
        return True
    if edge.kind == "seq" and edge.target == branch_node_id and edge.source in {body_region_id, empty_region_id}:
        return True
    if edge.kind == "seq" and edge.source == body_region_id and edge.target in removed_node_ids:
        return True
    if edge.kind == "seq" and edge.source == empty_region_id and edge.target in removed_node_ids:
        return True
    if edge.kind == "seq" and edge.target in {body_region_id, empty_region_id} and edge.source in removed_node_ids:
        return True
    return False


def _close_region_boundaries(region: UHIRRegion) -> None:
    node_by_id = {node.id: node for node in region.nodes}
    source = next((node for node in region.nodes if node.opcode == "nop" and node.attributes.get("role") == "source"), None)
    sink = next((node for node in region.nodes if node.opcode == "nop" and node.attributes.get("role") == "sink"), None)
    if source is None or sink is None:
        return

    existing = {(edge.kind, edge.source, edge.target) for edge in region.edges}
    for node in region.nodes:
        if node.id in {source.id, sink.id}:
            continue
        has_predecessor = any(edge.kind == "data" and edge.target == node.id and edge.source in node_by_id for edge in region.edges)
        if not has_predecessor and ("data", source.id, node.id) not in existing:
            region.edges.append(UHIREdge("data", source.id, node.id))
            existing.add(("data", source.id, node.id))

        has_successor = any(edge.kind == "data" and edge.source == node.id and edge.target in node_by_id for edge in region.edges)
        if not has_successor and ("data", node.id, sink.id) not in existing:
            region.edges.append(UHIREdge("data", node.id, sink.id))
            existing.add(("data", node.id, sink.id))


def _dedupe_region_edges(region: UHIRRegion) -> None:
    seen: set[tuple[object, ...]] = set()
    unique: list[UHIREdge] = []
    for edge in region.edges:
        key = (edge.kind, edge.source, edge.target, tuple(sorted(edge.attributes.items())), edge.directed)
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    region.edges = unique


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


def _fresh_node_counter(design: UHIRDesign) -> int:
    max_id = -1
    for region in design.regions:
        for node in region.nodes:
            if not node.id.startswith("v"):
                continue
            suffix = node.id[1:]
            if suffix.isdigit():
                max_id = max(max_id, int(suffix))
    return max_id + 1
