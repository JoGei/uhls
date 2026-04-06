"""Hierarchical sched lowering entrypoint."""

from __future__ import annotations

from uhls.backend.uhir.model import (
    UHIRConstant,
    UHIRDesign,
    UHIREdge,
    UHIRNode,
    UHIRPort,
    UHIRRegion,
    UHIRRegionRef,
    UHIRSchedule,
    UHIRSourceMap,
)
from uhls.backend.uhir.timing import TimingBinary, TimingCall, TimingExpr, TimingVar

from .interfaces import SGUScheduleResult, SGUScheduler
from .registry import create_builtin_scheduler


def lower_alloc_to_sched(
    design: UHIRDesign,
    *,
    scheduler: SGUScheduler | None = None,
    algorithm: str | None = None,
) -> UHIRDesign:
    """Lower one alloc-stage µhIR design to sched-stage µhIR."""
    if design.stage != "alloc":
        raise ValueError(f"sched lowering expects alloc-stage µhIR input, got stage '{design.stage}'")

    flat_scheduler = _resolve_scheduler(scheduler=scheduler, algorithm=algorithm)

    scheduled = UHIRDesign(name=design.name, stage="sched")
    scheduled.inputs = [_clone_port(port) for port in design.inputs]
    scheduled.outputs = [_clone_port(port) for port in design.outputs]
    scheduled.constants = [_clone_constant(constant) for constant in design.constants]
    scheduled.schedule = UHIRSchedule("hierarchical")
    scheduled.regions = [_clone_sched_region(region) for region in design.regions if region.kind != "executability"]

    region_by_id = {region.id: region for region in scheduled.regions}
    child_to_parent_node, children_by_region = _build_hierarchy(region_by_id)
    _schedule_bottom_up(region_by_id, children_by_region, scheduler=flat_scheduler)
    _shift_regions_into_global_time(region_by_id, child_to_parent_node)
    return scheduled


def _resolve_scheduler(*, scheduler: SGUScheduler | None, algorithm: str | None) -> SGUScheduler:
    if scheduler is not None:
        if algorithm is not None:
            raise ValueError("sched lowering accepts either one scheduler instance or one algorithm name, not both")
        return scheduler
    return create_builtin_scheduler("asap" if algorithm is None else algorithm)


def _schedule_bottom_up(
    region_by_id: dict[str, UHIRRegion],
    children_by_region: dict[str, list[str]],
    *,
    scheduler: SGUScheduler,
) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(region_id: str) -> None:
        if region_id in visited:
            return
        if region_id in visiting:
            raise ValueError(f"hierarchical scheduling requires an acyclic SGU graph; found cycle at '{region_id}'")
        visiting.add(region_id)
        for child_id in children_by_region[region_id]:
            visit(child_id)
        visiting.remove(region_id)
        visited.add(region_id)
        _materialize_child_latencies(region_by_id[region_id], region_by_id)
        _apply_schedule_result(region_by_id[region_id], scheduler.schedule_sgu(region_by_id[region_id]))

    for region_id in region_by_id:
        visit(region_id)


def _build_hierarchy(
    region_by_id: dict[str, UHIRRegion],
) -> tuple[dict[str, tuple[str, str]], dict[str, list[str]]]:
    child_to_parent_node: dict[str, tuple[str, str]] = {}
    children_by_region = {region_id: [] for region_id in region_by_id}
    for region in region_by_id.values():
        for node in region.nodes:
            child_ids = _hierarchy_children(node)
            if node.opcode == "loop":
                pass
            elif len(child_ids) > 1:
                if node.opcode != "branch":
                    raise NotImplementedError(
                        f"sched lowering currently supports only call, branch, and loop hierarchy; "
                        f"region '{region.id}' node '{node.id}' uses unsupported '{node.opcode}' timing"
                    )
            for child_id in child_ids:
                if child_id not in region_by_id:
                    raise ValueError(f"region '{region.id}' references unknown child region '{child_id}'")
                existing = child_to_parent_node.get(child_id)
                if existing is not None and existing != (region.id, node.id):
                    owner_region, owner_node = existing
                    raise NotImplementedError(
                        f"sched lowering currently requires one tree-shaped hierarchy; child region '{child_id}' "
                        f"is shared by '{owner_region}/{owner_node}' and '{region.id}/{node.id}'"
                    )
                child_to_parent_node[child_id] = (region.id, node.id)
                children_by_region[region.id].append(child_id)
    return child_to_parent_node, children_by_region


def _materialize_child_latencies(region: UHIRRegion, region_by_id: dict[str, UHIRRegion]) -> None:
    for node in region.nodes:
        child_ids = _hierarchy_children(node)
        if not child_ids:
            continue
        children = [region_by_id[child_id] for child_id in child_ids]
        for child in children:
            if child.latency is None:
                raise ValueError(f"child region '{child.id}' must be scheduled before parent '{region.id}'")
        if node.opcode == "loop":
            node.attributes["delay"] = _loop_total_latency(node, children[0], region_by_id)
            node.attributes["ii"] = _loop_initiation_interval(node, children[0])
            if _static_trip_count(node) is None:
                continue_condition = _loop_continue_condition(children[0])
                _set_symbolic_hierarchy_contract(
                    node,
                    handshake="ready_done",
                    ready=_symbolic_ready_name(node),
                    continue_condition=continue_condition,
                    iterate_when=continue_condition,
                    exit_when=_negated_condition(continue_condition),
                )
            else:
                _set_static_hierarchy_contract(node)
            continue
        if node.opcode == "branch":
            node.attributes["delay"] = _branch_total_latency(children)
            if isinstance(node.attributes["delay"], TimingExpr):
                _set_symbolic_hierarchy_contract(node, branch_condition=_branch_condition(node))
            else:
                _set_static_hierarchy_contract(node)
            continue
        if node.opcode == "call":
            node.attributes["delay"] = _call_total_latency(node, children[0])
            node.attributes["ii"] = _call_initiation_interval(node, children[0])
            if isinstance(node.attributes["delay"], TimingExpr):
                _set_symbolic_hierarchy_contract(node, handshake="ready_done", ready=_symbolic_ready_name(node))
            else:
                _set_static_hierarchy_contract(node)
            continue
        node.attributes["delay"] = children[0].latency


def _apply_schedule_result(region: UHIRRegion, result: SGUScheduleResult) -> None:
    if result.region_id != region.id:
        raise ValueError(f"scheduler returned schedule for region '{result.region_id}', expected '{region.id}'")

    node_by_id = {node.id: node for node in region.nodes}
    missing = sorted(node_id for node_id in node_by_id if node_id not in result.node_starts)
    if missing:
        raise ValueError(f"scheduler did not assign start/end times for region '{region.id}' nodes: {', '.join(missing)}")

    extras = sorted(node_id for node_id in result.node_starts if node_id not in node_by_id)
    if extras:
        raise ValueError(f"scheduler returned unknown nodes for region '{region.id}': {', '.join(extras)}")

    for node_id, (start, end) in result.node_starts.items():
        if not isinstance(start, (int, TimingExpr)) or not isinstance(end, (int, TimingExpr)):
            raise ValueError(f"scheduler returned invalid start/end types for region '{region.id}' node '{node_id}'")
        if isinstance(start, int) and isinstance(end, int) and end < start:
            raise ValueError(f"scheduler returned end < start for region '{region.id}' node '{node_id}'")
        node_by_id[node_id].attributes["start"] = start
        node_by_id[node_id].attributes["end"] = end

    source_sink_ids = {
        node.id
        for node in region.nodes
        if node.opcode == "nop" and node.attributes.get("role") in {"source", "sink"}
    }
    interior_nodes = [node for node in region.nodes if node.id not in source_sink_ids]
    if not interior_nodes:
        region.steps = (0, 0) if result.steps is None else result.steps
        region.latency = result.latency
        region.initiation_interval = result.initiation_interval
        return

    if result.steps is not None:
        region.steps = result.steps
    elif all(isinstance(node.attributes["start"], int) and isinstance(node.attributes["end"], int) for node in interior_nodes):
        region.steps = (
            min(int(node.attributes["start"]) for node in interior_nodes),
            max(int(node.attributes["end"]) for node in interior_nodes),
        )
    else:
        region.steps = None
    region.latency = result.latency
    region.initiation_interval = result.initiation_interval


def _shift_regions_into_global_time(
    region_by_id: dict[str, UHIRRegion],
    child_to_parent_node: dict[str, tuple[str, str]],
) -> None:
    roots = [region_id for region_id in region_by_id if region_id not in child_to_parent_node]
    for region_id in roots:
        _shift_region(region_by_id[region_id], region_by_id, offset=0)


def _shift_region(region: UHIRRegion, region_by_id: dict[str, UHIRRegion], *, offset: int) -> None:
    if offset:
        for node in region.nodes:
            node.attributes["start"] = _timing_add(node.attributes["start"], offset)
            node.attributes["end"] = _timing_add(node.attributes["end"], offset)
        if region.steps is not None:
            region.steps = (_timing_add(region.steps[0], offset), _timing_add(region.steps[1], offset))

    for node in region.nodes:
        for child_id in _hierarchy_children(node):
            child_offset = 0 if node.opcode == "loop" else node.attributes["start"]
            _shift_region(region_by_id[child_id], region_by_id, offset=child_offset)


def _hierarchy_children(node: UHIRNode) -> list[str]:
    children: list[str] = []
    for key in ("child", "true_child", "false_child"):
        value = node.attributes.get(key)
        if isinstance(value, str) and value:
            children.append(value)
    return children


def _static_trip_count(node: UHIRNode) -> int | None:
    trip_count = node.attributes.get("static_trip_count")
    if isinstance(trip_count, int) and trip_count >= 0:
        return trip_count
    return None


def _static_loop_total_latency(loop_node: UHIRNode, header_region: UHIRRegion, region_by_id: dict[str, UHIRRegion]) -> int:
    trip_count = _static_trip_count(loop_node)
    if trip_count is None:
        raise ValueError(f"loop node '{loop_node.id}' requires static_trip_count")

    branch = next((node for node in header_region.nodes if node.opcode == "branch"), None)
    if branch is None:
        raise ValueError(f"loop region '{header_region.id}' is missing its header branch node")

    true_child = branch.attributes.get("true_child")
    false_child = branch.attributes.get("false_child")
    if not isinstance(true_child, str) or not isinstance(false_child, str):
        raise ValueError(f"loop header branch '{branch.id}' must declare true_child/false_child regions")

    header_latency = 0 if header_region.latency is None else header_region.latency
    body_region = region_by_id[true_child]
    exit_region = region_by_id[false_child]
    body_latency = 0 if body_region.latency is None else body_region.latency
    exit_latency = 0 if exit_region.latency is None else exit_region.latency

    iter_initiation_interval = header_latency
    iter_latency = header_latency
    iter_ramp_down = iter_latency - iter_initiation_interval

    loop_node.attributes["iter_latency"] = iter_latency
    loop_node.attributes["iter_initiation_interval"] = iter_initiation_interval
    loop_node.attributes["iter_ramp_down"] = iter_ramp_down
    return trip_count * iter_initiation_interval + iter_ramp_down


def _loop_total_latency(loop_node: UHIRNode, header_region: UHIRRegion, region_by_id: dict[str, UHIRRegion]) -> int | TimingExpr:
    trip_count = _static_trip_count(loop_node)
    if trip_count is not None:
        return _static_loop_total_latency(loop_node, header_region, region_by_id)
    for name in ("iter_latency", "iter_initiation_interval", "iter_ramp_down"):
        loop_node.attributes.pop(name, None)
    return _symbolic_delay_var(loop_node)


def _loop_initiation_interval(loop_node: UHIRNode, header_region: UHIRRegion) -> int | TimingExpr:
    static_iter_ii = loop_node.attributes.get("iter_initiation_interval")
    if isinstance(static_iter_ii, (int, TimingExpr)):
        return static_iter_ii
    if isinstance(header_region.initiation_interval, (int, TimingExpr)):
        return header_region.initiation_interval
    return _symbolic_ii_var(loop_node)


def _branch_total_latency(children: list[UHIRRegion]) -> int | TimingExpr:
    latencies = [child.latency for child in children]
    if all(isinstance(latency, int) for latency in latencies):
        return max(latency for latency in latencies if latency is not None)
    expr_args: list[int | TimingExpr] = []
    for index, latency in enumerate(latencies):
        if isinstance(latency, (int, TimingExpr)):
            expr_args.append(latency)
        else:
            expr_args.append(TimingVar(f"child_{index}_lat"))
    return TimingCall("max", tuple(expr_args))


def _call_total_latency(call_node: UHIRNode, child_region: UHIRRegion) -> int | TimingExpr:
    if isinstance(child_region.latency, int):
        return child_region.latency
    return _symbolic_delay_var(call_node)


def _call_initiation_interval(call_node: UHIRNode, child_region: UHIRRegion) -> int | TimingExpr:
    if isinstance(child_region.initiation_interval, (int, TimingExpr)):
        return child_region.initiation_interval
    return _symbolic_ii_var(call_node)


def _timing_add(left: int | TimingExpr, right: int | TimingExpr) -> int | TimingExpr:
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    if isinstance(left, int) and left == 0:
        return right
    if isinstance(right, int) and right == 0:
        return left
    return TimingBinary(left, "+", right)


def _symbolic_delay_var(node: UHIRNode) -> TimingExpr:
    return TimingVar(f"symb_delay_{node.id}")


def _symbolic_ii_var(node: UHIRNode) -> TimingExpr:
    return TimingVar(f"symb_ii_{node.id}")


def _symbolic_ready_name(node: UHIRNode) -> str:
    return f"symb_ready_{node.id}"


def _symbolic_done_name(node: UHIRNode) -> str:
    return f"symb_done_{node.id}"


def _set_static_hierarchy_contract(node: UHIRNode) -> None:
    node.attributes["timing"] = "static"
    for name in (
        "completion",
        "ready",
        "handshake",
        "branch_condition",
        "continue_condition",
        "iterate_when",
        "exit_when",
    ):
        node.attributes.pop(name, None)


def _set_symbolic_hierarchy_contract(
    node: UHIRNode,
    *,
    handshake: str | None = None,
    ready: str | None = None,
    branch_condition: str | None = None,
    continue_condition: str | None = None,
    iterate_when: str | None = None,
    exit_when: str | None = None,
) -> None:
    node.attributes["timing"] = "symbolic"
    node.attributes["completion"] = _symbolic_done_name(node)
    if handshake is not None:
        node.attributes["handshake"] = handshake
    else:
        node.attributes.pop("handshake", None)
    if ready is not None:
        node.attributes["ready"] = ready
    else:
        node.attributes.pop("ready", None)
    if branch_condition is not None:
        node.attributes["branch_condition"] = branch_condition
    else:
        node.attributes.pop("branch_condition", None)
    if continue_condition is not None:
        node.attributes["continue_condition"] = continue_condition
    else:
        node.attributes.pop("continue_condition", None)
    if iterate_when is not None:
        node.attributes["iterate_when"] = iterate_when
    else:
        node.attributes.pop("iterate_when", None)
    if exit_when is not None:
        node.attributes["exit_when"] = exit_when
    else:
        node.attributes.pop("exit_when", None)


def _branch_condition(node: UHIRNode) -> str | None:
    if len(node.operands) == 1:
        return node.operands[0]
    return None


def _loop_continue_condition(header_region: UHIRRegion) -> str | None:
    branch = next((node for node in header_region.nodes if node.opcode == "branch"), None)
    if branch is None:
        return None
    return _branch_condition(branch)


def _negated_condition(condition: str | None) -> str | None:
    if condition is None:
        return None
    return f"!{condition}"


def _clone_sched_region(region: UHIRRegion) -> UHIRRegion:
    cloned = UHIRRegion(id=region.id, kind=region.kind, parent=region.parent)
    cloned.region_refs = [UHIRRegionRef(ref.target) for ref in region.region_refs]
    cloned.nodes = [_clone_node(node) for node in region.nodes]
    cloned.edges = [UHIREdge(edge.kind, edge.source, edge.target, dict(edge.attributes), edge.directed) for edge in region.edges]
    cloned.mappings = [UHIRSourceMap(mapping.node_id, mapping.source_id) for mapping in region.mappings]
    return cloned


def _clone_node(node: UHIRNode) -> UHIRNode:
    attributes = dict(node.attributes)
    attributes.pop("start", None)
    attributes.pop("end", None)
    return UHIRNode(node.id, node.opcode, node.operands, node.result_type, attributes)


def _clone_port(port: UHIRPort) -> UHIRPort:
    return UHIRPort(port.direction, port.name, port.type)


def _clone_constant(constant: UHIRConstant) -> UHIRConstant:
    return UHIRConstant(constant.name, constant.value, constant.type)
