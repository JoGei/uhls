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
                if _static_trip_count(node) is None:
                    raise NotImplementedError(
                        f"sched lowering currently supports only statically bounded loop timing; "
                        f"region '{region.id}' node '{node.id}' is missing static_trip_count"
                    )
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
            node.attributes["delay"] = _static_loop_total_latency(node, children[0], region_by_id)
            continue
        if node.opcode == "branch":
            node.attributes["delay"] = max(child.latency or 0 for child in children)
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
        if end < start:
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
        region.steps = (0, 0)
        region.latency = result.latency
        region.initiation_interval = result.initiation_interval
        return

    region.steps = (
        min(int(node.attributes["start"]) for node in interior_nodes),
        max(int(node.attributes["end"]) for node in interior_nodes),
    )
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
            node.attributes["start"] = int(node.attributes["start"]) + offset
            node.attributes["end"] = int(node.attributes["end"]) + offset
        if region.steps is not None:
            region.steps = (region.steps[0] + offset, region.steps[1] + offset)

    for node in region.nodes:
        for child_id in _hierarchy_children(node):
            child_offset = 0 if node.opcode == "loop" else int(node.attributes["start"])
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
