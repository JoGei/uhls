"""Static-fact inference on seq-stage µhIR."""

from __future__ import annotations

from copy import deepcopy
from math import gcd

from uhls.backend.hls.uhir.model import UHIRDesign, UHIRNode, UHIRRegion


class InferStaticPass:
    """Infer conservative static facts for seq-stage µhIR."""

    name = "infer_static"

    def run(self, ir: UHIRDesign) -> UHIRDesign:
        if ir.stage != "seq":
            raise ValueError(f"infer_static expects seq-stage µhIR, got stage '{ir.stage}'")
        result = deepcopy(ir)
        region_by_id = {region.id: region for region in result.regions}
        for region in result.regions:
            for node in region.nodes:
                if node.opcode != "loop":
                    continue
                child_id = node.attributes.get("child")
                if not isinstance(child_id, str):
                    continue
                trip_count = _infer_static_loop_trip_count(region_by_id, child_id)
                if trip_count is None:
                    node.attributes.pop("static_trip_count", None)
                    continue
                node.attributes["static_trip_count"] = trip_count
        return result


def _infer_static_loop_trip_count(region_by_id: dict[str, UHIRRegion], loop_region_id: str) -> int | None:
    loop_region = region_by_id.get(loop_region_id)
    if loop_region is None or loop_region.kind != "loop":
        return None

    branch = next((node for node in loop_region.nodes if node.opcode == "branch"), None)
    if branch is None or len(branch.operands) != 1:
        return None

    compare = _producer_by_name(loop_region).get(branch.operands[0])
    if compare is None or compare.opcode not in {"lt", "le", "gt", "ge"} or len(compare.operands) != 2:
        return None

    induction_var, compare_opcode, bound = _normalize_trip_count_compare(region_by_id, loop_region, compare)
    if induction_var is None or compare_opcode is None or bound is None:
        return None

    phi = _producer_by_name(loop_region).get(induction_var)
    if phi is None or phi.opcode != "phi" or len(phi.operands) != 2:
        return None

    body_region_id = branch.attributes.get("true_child")
    if not isinstance(body_region_id, str):
        return None
    body_region = region_by_id.get(body_region_id)
    if body_region is None:
        return None

    body_outputs = set(_producer_by_name(body_region))
    outside_incoming = next((operand for operand in phi.operands if operand not in body_outputs), None)
    backedge_incoming = next((operand for operand in phi.operands if operand in body_outputs), None)
    if outside_incoming is None or backedge_incoming is None:
        return None

    init = _resolve_constant_in_region(region_by_id, loop_region.id, outside_incoming)
    if init is None:
        return None

    update = _resolve_induction_update(region_by_id, body_region.id, backedge_incoming)
    if update is None:
        canonical = _resolve_induction_progression(region_by_id, body_region.id, backedge_incoming)
        if canonical is None:
            return None
        update_var, steps = canonical
        if update_var != induction_var or not steps or any(step_value == 0 for step_value in steps):
            return None
        if len(steps) == 1:
            only_step = next(iter(steps))
            return _compute_trip_count(init, bound, only_step, compare_opcode)
        return _compute_canonicalized_trip_count(init, bound, steps, compare_opcode)
    update_var, step = update
    if update_var != induction_var or step == 0:
        canonical = _resolve_induction_progression(region_by_id, body_region.id, backedge_incoming)
        if canonical is None:
            return None
        update_var, steps = canonical
        if update_var != induction_var or not steps or any(step_value == 0 for step_value in steps):
            return None
        if len(steps) == 1:
            only_step = next(iter(steps))
            return _compute_trip_count(init, bound, only_step, compare_opcode)
        return _compute_canonicalized_trip_count(init, bound, steps, compare_opcode)

    return _compute_trip_count(init, bound, step, compare_opcode)


def _producer_by_name(region: UHIRRegion) -> dict[str, UHIRNode]:
    produced = {node.id: node for node in region.nodes}
    for mapping in region.mappings:
        node = next((candidate for candidate in region.nodes if candidate.id == mapping.node_id), None)
        if node is not None:
            produced[mapping.source_id] = node
    return produced


def _normalize_trip_count_compare(
    region_by_id: dict[str, UHIRRegion],
    region: UHIRRegion,
    compare: UHIRNode,
) -> tuple[str | None, str | None, int | None]:
    lhs_name = _variable_name(compare.operands[0])
    rhs_name = _variable_name(compare.operands[1])
    lhs_const = _resolve_constant_in_region(region_by_id, region.id, compare.operands[0])
    rhs_const = _resolve_constant_in_region(region_by_id, region.id, compare.operands[1])

    if lhs_name is not None and rhs_const is not None:
        return lhs_name, compare.opcode, rhs_const
    if rhs_name is not None and lhs_const is not None:
        return rhs_name, _flip_compare_opcode(compare.opcode), lhs_const
    return None, None, None


def _flip_compare_opcode(opcode: str) -> str | None:
    return {
        "lt": "gt",
        "le": "ge",
        "gt": "lt",
        "ge": "le",
    }.get(opcode)


def _resolve_induction_update(region_by_id: dict[str, UHIRRegion], region_id: str, value: str) -> tuple[str, int] | None:
    region = region_by_id[region_id]
    node = _producer_by_name(region).get(value)
    if node is None:
        return None
    if node.opcode == "mov" and node.operands:
        return _resolve_induction_update(region_by_id, region_id, node.operands[0])
    if node.opcode not in {"add", "sub"} or len(node.operands) != 2:
        return None

    lhs_name = _variable_name(node.operands[0])
    rhs_name = _variable_name(node.operands[1])
    lhs_const = _resolve_constant_in_region(region_by_id, region_id, node.operands[0])
    rhs_const = _resolve_constant_in_region(region_by_id, region_id, node.operands[1])

    if lhs_name is not None and rhs_const is not None:
        delta = rhs_const if node.opcode == "add" else -rhs_const
        return lhs_name, delta
    if rhs_name is not None and lhs_const is not None and node.opcode == "add":
        return rhs_name, lhs_const
    return None


def _resolve_induction_progression(
    region_by_id: dict[str, UHIRRegion],
    region_id: str,
    value: str,
    *,
    _seen: set[tuple[str, str]] | None = None,
) -> tuple[str, set[int]] | None:
    seen = set() if _seen is None else set(_seen)
    key = (region_id, value)
    if key in seen:
        return None
    seen.add(key)

    region = region_by_id[region_id]
    node = _producer_by_name(region).get(value)
    if node is None:
        if region.parent is None:
            return None
        return _resolve_induction_progression(region_by_id, region.parent, value, _seen=seen)

    if node.opcode == "mov" and node.operands:
        return _resolve_induction_progression(region_by_id, region_id, node.operands[0], _seen=seen)

    if node.opcode == "phi" and node.operands:
        base_var: str | None = None
        steps: set[int] = set()
        for operand in node.operands:
            resolved = _resolve_induction_progression(region_by_id, region_id, operand, _seen=seen)
            if resolved is None:
                return None
            operand_var, operand_steps = resolved
            if base_var is None:
                base_var = operand_var
            elif operand_var != base_var:
                return None
            steps.update(operand_steps)
        if base_var is None or not steps:
            return None
        return base_var, steps

    if node.opcode not in {"add", "sub"} or len(node.operands) != 2:
        return None

    lhs_progression = _operand_progression(region_by_id, region_id, node.operands[0], seen)
    rhs_progression = _operand_progression(region_by_id, region_id, node.operands[1], seen)
    lhs_const = _resolve_constant_in_region(region_by_id, region_id, node.operands[0])
    rhs_const = _resolve_constant_in_region(region_by_id, region_id, node.operands[1])

    if lhs_progression is not None and rhs_const is not None:
        base_var, lhs_steps = lhs_progression
        delta = rhs_const if node.opcode == "add" else -rhs_const
        return base_var, {step + delta for step in lhs_steps}
    if rhs_progression is not None and lhs_const is not None and node.opcode == "add":
        base_var, rhs_steps = rhs_progression
        return base_var, {lhs_const + step for step in rhs_steps}
    return None


def _operand_progression(
    region_by_id: dict[str, UHIRRegion],
    region_id: str,
    operand: str,
    seen: set[tuple[str, str]],
) -> tuple[str, set[int]] | None:
    region = region_by_id[region_id]
    direct_name = _variable_name(operand)
    if direct_name is not None:
        local_producer = _producer_by_name(region).get(direct_name)
        if local_producer is not None and local_producer.id != direct_name:
            return _resolve_induction_progression(region_by_id, region_id, direct_name, _seen=seen)
        return direct_name, {0}
    return _resolve_induction_progression(region_by_id, region_id, operand, _seen=seen)


def _resolve_constant_in_region(region_by_id: dict[str, UHIRRegion], region_id: str, value: str) -> int | None:
    literal = _literal_value(value)
    if literal is not None:
        return literal
    region = region_by_id[region_id]
    node = _producer_by_name(region).get(value)
    if node is None:
        if region.parent is None:
            return None
        return _resolve_constant_in_region(region_by_id, region.parent, value)
    if node.opcode == "const" and node.operands:
        return _literal_value(node.operands[0])
    if node.opcode == "mov" and node.operands:
        return _resolve_constant_in_region(region_by_id, region_id, node.operands[0])
    return None


def _literal_value(text: str) -> int | None:
    value_text = text.split(":", 1)[0].strip()
    if not value_text:
        return None
    if value_text[0] not in "-0123456789":
        return None
    try:
        return int(value_text)
    except ValueError:
        return None


def _variable_name(text: str) -> str | None:
    if not text:
        return None
    if text[0].isalpha() or text[0] in {"_", "%"}:
        return text
    return None


def _compute_trip_count(init: int, bound: int, step: int, compare_opcode: str) -> int | None:
    if step > 0 and compare_opcode in {"lt", "le"}:
        distance = bound - init if compare_opcode == "lt" else bound - init + 1
        if distance <= 0:
            return 0
        return (distance + step - 1) // step
    if step < 0 and compare_opcode in {"gt", "ge"}:
        stride = -step
        distance = init - bound if compare_opcode == "gt" else init - bound + 1
        if distance <= 0:
            return 0
        return (distance + stride - 1) // stride
    return None


def _compute_canonicalized_trip_count(
    init: int,
    bound: int,
    steps: set[int],
    compare_opcode: str,
) -> int | None:
    ordered = sorted(steps)
    if not ordered:
        return None
    same_sign = all(step > 0 for step in ordered) or all(step < 0 for step in ordered)
    if not same_sign:
        return None

    base_step = abs(ordered[0])
    for step in ordered[1:]:
        base_step = gcd(base_step, abs(step))
    if base_step == 0:
        return None

    normalized = sorted(abs(step) // base_step for step in ordered)
    if normalized != list(range(1, normalized[-1] + 1)):
        return None

    signed_base = base_step if ordered[0] > 0 else -base_step
    original_trip_count = _compute_trip_count(init, bound, signed_base, compare_opcode)
    if original_trip_count is None:
        return None
    factor = normalized[-1]
    return (original_trip_count + factor - 1) // factor
