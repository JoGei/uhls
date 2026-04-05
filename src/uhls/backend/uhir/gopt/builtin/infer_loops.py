"""Explicit-loop inference annotations on seq-stage µhIR."""

from __future__ import annotations

from copy import deepcopy

from uhls.backend.uhir.model import UHIRDesign, UHIREdge, UHIRNode, UHIRRegion
from uhls.backend.uhir.gopt.loops import collect_explicit_loops, collect_loop_candidates


class InferLoopsPass:
    """Annotate branch-based or explicit loop hierarchy conservatively."""

    name = "infer_loops"

    def run(self, ir: UHIRDesign) -> UHIRDesign:
        if ir.stage != "seq":
            raise ValueError(f"infer_loops expects seq-stage µhIR, got stage '{ir.stage}'")

        result = deepcopy(ir)
        explicit_loops = collect_explicit_loops(result)
        candidates = collect_loop_candidates(result)

        for candidate in candidates:
            candidate.branch_node.attributes["loop_id"] = candidate.loop_id
            candidate.branch_node.attributes["loop_header"] = True
            for node in candidate.parent_region.nodes:
                if node.id not in candidate.header_node_ids:
                    continue
                node.attributes["loop_member"] = candidate.loop_id
                node.attributes["loop_role"] = "header"
            for node in candidate.body_region.nodes:
                node.attributes["loop_member"] = candidate.loop_id
                node.attributes["loop_role"] = "body"
            for node in candidate.empty_region.nodes:
                node.attributes["loop_member"] = candidate.loop_id
                node.attributes["loop_role"] = "exit"
            for edge in candidate.parent_region.edges:
                if edge.kind == "seq" and edge.source == candidate.branch_node.id and edge.target == candidate.body_region.id:
                    edge.attributes["loop_id"] = candidate.loop_id
                    edge.attributes["loop_role"] = "body"
                elif edge.kind == "seq" and edge.source == candidate.branch_node.id and edge.target == candidate.empty_region.id:
                    edge.attributes["loop_id"] = candidate.loop_id
                    edge.attributes["loop_role"] = "exit"
                elif edge.kind == "seq" and edge.source == candidate.body_region.id and edge.target == candidate.branch_node.id:
                    edge.attributes["loop_id"] = candidate.loop_id
                    edge.attributes["loop_backedge"] = True

        for info in explicit_loops:
            _annotate_region_nodes(info.header_region, info.loop_id, "header")
            info.loop_node.attributes["loop_id"] = info.loop_id
            if info.header_branch is not None:
                info.header_branch.attributes["loop_id"] = info.loop_id
                info.header_branch.attributes["loop_header"] = True
            if info.body_region is not None:
                _annotate_region_nodes(info.body_region, info.loop_id, "body")
            if info.empty_region is not None:
                _annotate_region_nodes(info.empty_region, info.loop_id, "exit")

            _annotate_edge(info.entry_edge, info.loop_id, "header")
            _annotate_edge(info.return_edge, info.loop_id, "header")
            _annotate_edge(info.body_edge, info.loop_id, "body")
            _annotate_edge(info.exit_edge, info.loop_id, "exit")
            if info.backedge is not None:
                info.backedge.attributes["loop_id"] = info.loop_id
                info.backedge.attributes["loop_backedge"] = True

        return result


def _annotate_region_nodes(region: UHIRRegion, loop_id: str, role: str) -> None:
    for node in region.nodes:
        node.attributes["loop_member"] = loop_id
        node.attributes["loop_role"] = role


def _annotate_edge(edge: UHIREdge | None, loop_id: str, role: str) -> None:
    if edge is None:
        return
    edge.attributes["loop_id"] = loop_id
    edge.attributes["loop_role"] = role
