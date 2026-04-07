"""Predicate-folding cleanup on seq-stage µhIR."""

from __future__ import annotations

from copy import deepcopy

from uhls.backend.hls.uhir.model import UHIREdge, UHIRDesign, UHIRNode


class FoldPredicatesPass:
    """Fold redundant complementary predicated dataflow."""

    name = "fold_predicates"

    def run(self, ir: UHIRDesign) -> UHIRDesign:
        if ir.stage != "seq":
            raise ValueError(f"fold_predicates expects seq-stage µhIR, got stage '{ir.stage}'")
        result = deepcopy(ir)
        changed = True
        while changed:
            changed = False
            for region in result.regions:
                if _fold_region(region):
                    changed = True
                    break
        return result


def _fold_region(region) -> bool:
    producer_by_name = _producer_by_name(region)
    for node in list(region.nodes):
        if node.opcode != "sel" or len(node.operands) != 3:
            continue
        condition, true_value, false_value = node.operands
        if true_value == false_value:
            _replace_select_with_alias(region, node, true_value)
            return True
        true_producer = producer_by_name.get(true_value)
        false_producer = producer_by_name.get(false_value)
        if true_producer is None or false_producer is None:
            continue
        if not _predicated_pair_matches(condition, true_producer, false_producer):
            continue
        if not _same_pure_expression(true_producer, false_producer):
            continue
        if not _only_used_by(region, true_producer.id, node.id):
            continue
        if not _only_used_by(region, false_producer.id, node.id):
            continue
        _replace_select_with_equivalent_op(region, node, true_producer, false_producer)
        return True
    return False


def _producer_by_name(region) -> dict[str, UHIRNode]:
    produced = {node.id: node for node in region.nodes}
    for mapping in region.mappings:
        node = next((candidate for candidate in region.nodes if candidate.id == mapping.node_id), None)
        if node is not None:
            produced[mapping.source_id] = node
    return produced


def _predicated_pair_matches(condition: str, true_node: UHIRNode, false_node: UHIRNode) -> bool:
    true_pred = true_node.attributes.get("pred")
    false_pred = false_node.attributes.get("pred")
    if not isinstance(true_pred, str) or not isinstance(false_pred, str):
        return False
    return true_pred == condition and false_pred == f"!{condition}"


def _same_pure_expression(lhs: UHIRNode, rhs: UHIRNode) -> bool:
    if lhs.opcode != rhs.opcode or lhs.operands != rhs.operands or lhs.result_type != rhs.result_type:
        return False
    lhs_attrs = {name: value for name, value in lhs.attributes.items() if name != "pred"}
    rhs_attrs = {name: value for name, value in rhs.attributes.items() if name != "pred"}
    return lhs_attrs == rhs_attrs and _is_pure_opcode(lhs.opcode)


def _is_pure_opcode(opcode: str) -> bool:
    return opcode not in {"nop", "branch", "loop", "call", "ret", "store", "print", "load", "phi", "sel"}


def _only_used_by(region, producer_id: str, consumer_id: str) -> bool:
    consumers = {edge.target for edge in region.edges if edge.kind == "data" and edge.source == producer_id}
    return consumers == {consumer_id}


def _replace_select_with_alias(region, select_node: UHIRNode, aliased_value: str) -> None:
    select_node.opcode = "mov"
    select_node.operands = (aliased_value,)
    select_node.attributes.pop("incoming", None)
    region.edges = [edge for edge in region.edges if not (edge.kind == "data" and edge.target == select_node.id)]
    _wire_operands(region, select_node)


def _replace_select_with_equivalent_op(region, select_node: UHIRNode, true_node: UHIRNode, false_node: UHIRNode) -> None:
    select_node.opcode = true_node.opcode
    select_node.operands = true_node.operands
    select_node.result_type = true_node.result_type
    select_node.attributes = {name: value for name, value in true_node.attributes.items() if name != "pred"}
    region.edges = [
        edge
        for edge in region.edges
        if edge.source not in {true_node.id, false_node.id}
        and edge.target not in {true_node.id, false_node.id}
        and not (edge.kind == "data" and edge.target == select_node.id)
    ]
    region.nodes = [node for node in region.nodes if node.id not in {true_node.id, false_node.id}]
    region.mappings = [mapping for mapping in region.mappings if mapping.node_id not in {true_node.id, false_node.id}]
    _wire_operands(region, select_node)


def _wire_operands(region, node: UHIRNode) -> None:
    source_id = next(
        (candidate.id for candidate in region.nodes if candidate.opcode == "nop" and candidate.attributes.get("role") == "source"),
        None,
    )
    producer_by_name = _producer_by_name(region)
    seen: set[tuple[str, str]] = set()
    for operand in node.operands:
        producer = producer_by_name.get(operand)
        if producer is None:
            if source_id is None or ":" in operand or operand.startswith("!"):
                continue
            producer_id = source_id
        else:
            producer_id = producer.id
        pair = (producer_id, node.id)
        if pair in seen:
            continue
        seen.add(pair)
        region.edges.append(UHIREdge("data", producer_id, node.id))
