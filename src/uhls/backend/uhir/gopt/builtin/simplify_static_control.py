"""Static-control simplification on seq-stage µhIR."""

from __future__ import annotations

from copy import deepcopy

from uhls.backend.uhir.model import UHIRDesign, UHIREdge, UHIRNode


class SimplifyStaticControlPass:
    """Conservative static-control cleanup for seq-stage µhIR."""

    name = "simplify_static_control"

    def run(self, ir: UHIRDesign) -> UHIRDesign:
        if ir.stage != "seq":
            raise ValueError(f"simplify_static_control expects seq-stage µhIR, got stage '{ir.stage}'")
        result = deepcopy(ir)
        region_by_id = {region.id: region for region in result.regions}
        for region in result.regions:
            for node in region.nodes:
                if node.opcode != "loop":
                    continue
                if not isinstance(node.attributes.get("static_trip_count"), int):
                    continue
                child_id = node.attributes.get("child")
                if not isinstance(child_id, str):
                    continue
                child_region = region_by_id.get(child_id)
                if child_region is None or child_region.kind != "loop":
                    continue
                _simplify_static_loop_header(child_region)

        # TODO: Extend this pass beyond static counted loops once the desired
        # seq-stage canonical form is nailed down for statically decidable
        # branch hierarchy and later predication-oriented rewrites.
        return result


def _simplify_static_loop_header(region) -> None:
    branch = next((node for node in region.nodes if node.opcode == "branch"), None)
    if branch is None or len(branch.operands) != 1:
        return

    compare_name = branch.operands[0]
    compare_node = _mapped_producer(region, compare_name)
    if compare_node is None or compare_node.opcode not in {"lt", "le", "gt", "ge", "eq", "ne"}:
        return
    if not _is_only_used_by_static_branch(region, compare_node.id, branch.id):
        return

    incoming_edges = [
        edge
        for edge in region.edges
        if edge.kind == "data" and edge.target == compare_node.id and edge.source != branch.id
    ]
    region.nodes = [node for node in region.nodes if node.id != compare_node.id]
    region.edges = [
        edge
        for edge in region.edges
        if edge.source != compare_node.id and edge.target != compare_node.id
    ]
    region.mappings = [mapping for mapping in region.mappings if mapping.node_id != compare_node.id]
    branch.operands = ()
    for edge in incoming_edges:
        replacement = UHIREdge(edge.kind, edge.source, branch.id, dict(edge.attributes), directed=edge.directed)
        if not _has_edge(region.edges, replacement):
            region.edges.append(replacement)


def _mapped_producer(region, source_name: str) -> UHIRNode | None:
    for mapping in region.mappings:
        if mapping.source_id != source_name:
            continue
        return next((node for node in region.nodes if node.id == mapping.node_id), None)
    return next((node for node in region.nodes if node.id == source_name), None)


def _is_only_used_by_static_branch(region, producer_id: str, branch_id: str) -> bool:
    consumers = {
        edge.target
        for edge in region.edges
        if edge.kind == "data" and edge.source == producer_id
    }
    return consumers == {branch_id}


def _has_edge(edges: list[UHIREdge], candidate: UHIREdge) -> bool:
    return any(
        edge.kind == candidate.kind
        and edge.source == candidate.source
        and edge.target == candidate.target
        and edge.attributes == candidate.attributes
        and edge.directed == candidate.directed
        for edge in edges
    )
