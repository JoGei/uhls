"""Built-in ASAP scheduling for one flat SGU."""

from __future__ import annotations

from uhls.backend.uhir.model import UHIRNode, UHIRRegion
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

        starts: dict[str, tuple[int, int]] = {}
        for node in topo_order:
            node_id = node.id
            start = 0
            for pred_id in incoming[node_id]:
                pred = node_by_id[pred_id]
                start = max(start, starts[pred_id][0] + _completion_delay(pred))
            starts[node_id] = (start, _scheduled_end(node, start))

        source_sink_ids = {
            node.id
            for node in region.nodes
            if node.opcode == "nop" and node.attributes.get("role") in {"source", "sink"}
        }
        interior_nodes = [node for node in region.nodes if node.id not in source_sink_ids]
        latency = 0 if not interior_nodes else max(starts[node.id][1] + 1 for node in interior_nodes)
        return SGUScheduleResult(
            region_id=region.id,
            node_starts=starts,
            latency=latency,
            initiation_interval=None,
        )


def _completion_delay(node: UHIRNode) -> int:
    delay = node.attributes.get("delay")
    if not isinstance(delay, int) or delay < 0:
        raise ValueError(f"node '{node.id}' must declare one non-negative integer delay")
    return delay


def _scheduled_end(node: UHIRNode, start: int) -> int:
    delay = _completion_delay(node)
    if delay == 0:
        return start
    return start + delay - 1
