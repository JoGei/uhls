"""Predication-oriented control simplification on seq-stage µhIR."""

from __future__ import annotations

from copy import deepcopy

from uhls.backend.hls.uhir.model import UHIREdge, UHIRDesign, UHIRNode, UHIRSourceMap


class PredicatePass:
    """Conservative partial if-conversion for branch-export dataflow."""

    name = "predicate"

    def run(self, ir: UHIRDesign) -> UHIRDesign:
        if ir.stage != "seq":
            raise ValueError(f"predicate expects seq-stage µhIR, got stage '{ir.stage}'")
        result = deepcopy(ir)
        changed = True
        while changed:
            changed = False
            region_by_id = {region.id: region for region in result.regions}
            for region in result.regions:
                for branch in list(region.nodes):
                    if branch.opcode != "branch":
                        continue
                    if _predicate_branch(result, region_by_id, region, branch):
                        changed = True
                        break
                if changed:
                    break
        return result


def _predicate_branch(design: UHIRDesign, region_by_id: dict[str, object], parent_region, branch_node: UHIRNode) -> bool:
    condition = branch_node.operands[0] if len(branch_node.operands) == 1 else None
    if not isinstance(condition, str) or not condition:
        return False

    true_child = branch_node.attributes.get("true_child")
    false_child = branch_node.attributes.get("false_child")
    if not isinstance(true_child, str) or not isinstance(false_child, str):
        return False
    true_region = region_by_id.get(true_child)
    false_region = region_by_id.get(false_child)
    if true_region is None or false_region is None:
        return False
    if true_region.parent != parent_region.id or false_region.parent != parent_region.id:
        return False
    if true_region.kind not in {"basicblock", "empty"} or false_region.kind not in {"basicblock", "empty"}:
        return False

    join_phis = _join_phi_nodes(parent_region, branch_node.id)
    if not join_phis:
        return False

    true_input_label = branch_node.attributes.get("true_input_label")
    false_input_label = branch_node.attributes.get("false_input_label")
    if not isinstance(true_input_label, str) or not isinstance(false_input_label, str):
        return False

    true_exports: set[str] = set()
    false_exports: set[str] = set()
    phi_arm_values: dict[str, tuple[str, str]] = {}
    for phi in join_phis:
        resolved = _resolve_phi_arms(phi, true_input_label=true_input_label, false_input_label=false_input_label)
        if resolved is None:
            return False
        true_value, false_value = resolved
        phi_arm_values[phi.id] = (true_value, false_value)
        if _produced_in_region(true_region, true_value):
            true_exports.add(true_value)
        if _produced_in_region(false_region, false_value):
            false_exports.add(false_value)

    true_slice = _slice_region_exports(true_region, true_exports)
    false_slice = _slice_region_exports(false_region, false_exports)
    if true_slice is None or false_slice is None:
        return False
    if not _slice_consumes_region(true_region, true_slice):
        return False
    if not _slice_consumes_region(false_region, false_slice):
        return False

    insert_at = parent_region.nodes.index(branch_node)
    true_nodes, true_mappings, true_edges = _materialize_predicated_nodes(parent_region, true_region, true_slice, predicate=condition)
    false_nodes, false_mappings, false_edges = _materialize_predicated_nodes(parent_region, false_region, false_slice, predicate=f"!{condition}")
    parent_region.nodes[insert_at:insert_at] = [*true_nodes, *false_nodes]
    parent_region.mappings.extend([*true_mappings, *false_mappings])
    parent_region.edges.extend([*true_edges, *false_edges])

    for phi in join_phis:
        true_value, false_value = phi_arm_values[phi.id]
        phi.opcode = "sel"
        phi.operands = (condition, true_value, false_value)
        phi.attributes.pop("incoming", None)
        parent_region.edges = [
            edge for edge in parent_region.edges if not (edge.kind == "data" and edge.target == phi.id)
        ]
        _wire_parent_operands(parent_region, phi)

    parent_region.nodes = [node for node in parent_region.nodes if node.id != branch_node.id]
    parent_region.edges = [
        edge
        for edge in parent_region.edges
        if edge.source != branch_node.id
        and edge.target != branch_node.id
        and edge.source not in {true_region.id, false_region.id}
        and edge.target not in {true_region.id, false_region.id}
    ]
    parent_region.region_refs = [ref for ref in parent_region.region_refs if ref.target not in {true_region.id, false_region.id}]
    design.regions = [region for region in design.regions if region.id not in {true_region.id, false_region.id}]
    return True


def _join_phi_nodes(region, branch_id: str) -> list[UHIRNode]:
    branch_users = {edge.target for edge in region.edges if edge.kind == "data" and edge.source == branch_id}
    return [node for node in region.nodes if node.id in branch_users and node.opcode == "phi"]


def _resolve_phi_arms(phi: UHIRNode, *, true_input_label: str, false_input_label: str) -> tuple[str, str] | None:
    incoming = phi.attributes.get("incoming")
    if not isinstance(incoming, tuple) or len(incoming) != len(phi.operands):
        return None
    value_by_label = {label: value for label, value in zip(incoming, phi.operands, strict=False)}
    if true_input_label not in value_by_label or false_input_label not in value_by_label:
        return None
    return value_by_label[true_input_label], value_by_label[false_input_label]


def _produced_in_region(region, value: str) -> bool:
    producer_by_name = _producer_by_name(region)
    producer = producer_by_name.get(value)
    return producer is not None and any(node.id == producer.id for node in region.nodes)


def _slice_region_exports(region, exports: set[str]) -> list[str] | None:
    producer_by_name = _producer_by_name(region)
    required_ids: set[str] = set()
    stack = [producer_by_name[name] for name in sorted(exports) if name in producer_by_name]
    while stack:
        node = stack.pop()
        if node.id in required_ids:
            continue
        if not _is_predicatable_opcode(node.opcode):
            return None
        required_ids.add(node.id)
        for operand in node.operands:
            producer = producer_by_name.get(operand)
            if producer is not None and any(candidate.id == producer.id for candidate in region.nodes):
                stack.append(producer)
    return [node.id for node in region.nodes if node.id in required_ids]


def _slice_consumes_region(region, slice_node_ids: list[str]) -> bool:
    real_ids = {
        node.id
        for node in region.nodes
        if not (node.opcode == "nop" and node.attributes.get("role") in {"source", "sink"})
    }
    return real_ids == set(slice_node_ids)


def _is_predicatable_opcode(opcode: str) -> bool:
    return opcode not in {"nop", "branch", "loop", "call", "ret", "store", "print", "load", "phi", "sel"}


def _materialize_predicated_nodes(parent_region, child_region, slice_node_ids: list[str], *, predicate: str):
    producer_by_name = _producer_by_name(parent_region)
    source_id = next(
        (node.id for node in parent_region.nodes if node.opcode == "nop" and node.attributes.get("role") == "source"),
        None,
    )
    child_nodes = {node.id: node for node in child_region.nodes}
    mapping_by_node: dict[str, list[UHIRSourceMap]] = {}
    for mapping in child_region.mappings:
        if mapping.node_id in child_nodes:
            mapping_by_node.setdefault(mapping.node_id, []).append(mapping)

    moved_nodes: list[UHIRNode] = []
    moved_mappings: list[UHIRSourceMap] = []
    moved_edges: list[UHIREdge] = []
    for node_id in slice_node_ids:
        node = child_nodes[node_id]
        node.attributes["pred"] = predicate
        moved_nodes.append(node)
        moved_mappings.extend(mapping_by_node.get(node.id, []))
        for operand in node.operands:
            producer = producer_by_name.get(operand)
            if producer is None:
                if source_id is not None and ":" not in operand:
                    moved_edges.append(UHIREdge("data", source_id, node.id))
                continue
            moved_edges.append(UHIREdge("data", producer.id, node.id))
        if not node.operands and source_id is not None:
            moved_edges.append(UHIREdge("data", source_id, node.id))
        producer_by_name[node.id] = node
        for mapping in mapping_by_node.get(node.id, []):
            producer_by_name[mapping.source_id] = node
    return moved_nodes, moved_mappings, _dedupe_edges(moved_edges)


def _wire_parent_operands(region, node: UHIRNode) -> None:
    producer_by_name = _producer_by_name(region)
    source_id = next(
        (candidate.id for candidate in region.nodes if candidate.opcode == "nop" and candidate.attributes.get("role") == "source"),
        None,
    )
    existing = {(edge.kind, edge.source, edge.target) for edge in region.edges}
    for operand in node.operands:
        producer = producer_by_name.get(operand)
        if producer is None:
            if source_id is None or ":" in operand or operand.startswith("!"):
                continue
            candidate = ("data", source_id, node.id)
            if candidate in existing:
                continue
            existing.add(candidate)
            region.edges.append(UHIREdge("data", source_id, node.id))
            continue
        candidate = ("data", producer.id, node.id)
        if candidate in existing:
            continue
        existing.add(candidate)
        region.edges.append(UHIREdge("data", producer.id, node.id))


def _producer_by_name(region) -> dict[str, UHIRNode]:
    produced = {node.id: node for node in region.nodes}
    for mapping in region.mappings:
        node = next((candidate for candidate in region.nodes if candidate.id == mapping.node_id), None)
        if node is not None:
            produced[mapping.source_id] = node
    return produced


def _dedupe_edges(edges: list[UHIREdge]) -> list[UHIREdge]:
    seen: set[tuple[object, ...]] = set()
    kept: list[UHIREdge] = []
    for edge in edges:
        key = (edge.kind, edge.source, edge.target, tuple(sorted(edge.attributes.items())), edge.directed)
        if key in seen:
            continue
        seen.add(key)
        kept.append(edge)
    return kept
