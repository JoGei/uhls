"""Built-in ASAP scheduling for one flat SGU."""

from __future__ import annotations

from uhls.backend.hls.uhir.model import TimingValue, UHIRNode, UHIRRegion
from uhls.backend.hls.uhir.timing import TimingBinary, TimingCall, TimingExpr, simplify_timing_expr
from uhls.utils.graph import topological_sort

from ..interfaces import SGUScheduleResult, SGUSchedulerBase


class ASAPScheduler(SGUSchedulerBase):
    """Schedule one allocated SGU as early as possible."""

    def schedule_sgu(self, region: UHIRRegion) -> SGUScheduleResult:
        self.validate_sgu_region(region)
        node_by_id = {node.id: node for node in region.nodes}
        incoming: dict[str, list[str]] = {node.id: [] for node in region.nodes}
        outgoing: dict[str, list[str]] = {node.id: [] for node in region.nodes}

        for edge in self.get_data_edges(region):
            outgoing[edge.source].append(edge.target)
            incoming[edge.target].append(edge.source)

        topo_order = topological_sort(
            region.nodes,
            lambda current: (node_by_id[succ_id] for succ_id in outgoing[current.id]),
            key=lambda current: current.id,
            cycle_error=lambda _: ValueError(f"region '{region.id}' is cyclic and cannot be ALAP-scheduled"),
        )

        starts: dict[str, tuple[TimingValue, TimingValue]] = {}
        for node in topo_order:
            node_id = node.id
            start: TimingValue = 0
            for pred_id in incoming[node_id]:
                pred = node_by_id[pred_id]
                start = _timing_max(start, _completion_ready_time(starts[pred_id][0], pred))
            starts[node_id] = (start, _scheduled_end(node, start))

        source_sink_ids = {
            node.id
            for node in region.nodes
            if node.opcode == "nop" and node.attributes.get("role") in {"source", "sink"}
        }
        interior_nodes = [node for node in region.nodes if node.id not in source_sink_ids]
        latency: TimingValue = 0
        for node in interior_nodes:
            latency = _timing_max(latency, _timing_add(starts[node.id][1], 1))
        return SGUScheduleResult(
            region_id=region.id,
            node_starts=starts,
            latency=latency,
            initiation_interval=None,
        )


def _completion_delay(node: UHIRNode) -> TimingValue:
    delay = node.attributes.get("delay")
    if isinstance(delay, int):
        if delay < 0:
            raise ValueError(f"node '{node.id}' must declare one non-negative integer delay")
        return delay
    if isinstance(delay, TimingExpr):
        return delay
    raise ValueError(f"node '{node.id}' must declare one non-negative integer delay")
    return delay


def _scheduled_end(node: UHIRNode, start: TimingValue) -> TimingValue:
    delay = _completion_delay(node)
    if delay == 0:
        return start
    return _timing_sub_one(_timing_add(start, delay))


def _completion_ready_time(start: TimingValue, node: UHIRNode) -> TimingValue:
    delay = _completion_delay(node)
    if delay == 0:
        return start
    return _timing_add(start, delay)


def _timing_add(left: TimingValue, right: TimingValue) -> TimingValue:
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    if isinstance(left, int) and left == 0:
        return right
    if isinstance(right, int) and right == 0:
        return left
    simplified = simplify_timing_expr(TimingBinary(left, "+", right))
    if isinstance(simplified, int):
        return simplified
    return simplified


def _timing_sub_one(value: TimingValue) -> TimingValue:
    if isinstance(value, int):
        return value - 1
    simplified = simplify_timing_expr(TimingBinary(value, "-", 1))
    if isinstance(simplified, int):
        return simplified
    return simplified


def _timing_max(left: TimingValue, right: TimingValue) -> TimingValue:
    if isinstance(left, int) and isinstance(right, int):
        return max(left, right)
    if left == right:
        return left
    simplified = simplify_timing_expr(TimingCall("max", (left, right)))
    if isinstance(simplified, int):
        return simplified
    return simplified
