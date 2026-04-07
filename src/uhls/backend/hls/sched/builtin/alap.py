"""Built-in ALAP scheduling for one flat SGU."""

from __future__ import annotations

from dataclasses import dataclass

from uhls.backend.hls.uhir.model import UHIRNode, UHIRRegion
from uhls.utils.graph import topological_sort

from ..interfaces import SGUScheduleResult, SGUSchedulerBase
from .asap import ASAPScheduler


@dataclass(slots=True)
class ALAPScheduler(SGUSchedulerBase):
    """Schedule one allocated SGU as late as possible under one latency bound."""

    sgu_latency_max: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.sgu_latency_max is None:
            raise ValueError("ALAPScheduler requires --sgu_latency_max")
        if not isinstance(self.sgu_latency_max, dict):
            raise ValueError("ALAPScheduler requires parsed sgu_latency_max scheduler arguments")

    def schedule_sgu(self, region: UHIRRegion) -> SGUScheduleResult:
        self.validate_sgu_region(region)
        sink = self.get_sink(region)
        max_latency = _resolve_region_latency_max(region, self.sgu_latency_max)
        asap_latency = ASAPScheduler().schedule_sgu(region).latency
        if max_latency < asap_latency:
            raise ValueError(
                f"ALAPScheduler cannot schedule region '{region.id}' with latency {max_latency}; "
                f"minimum feasible latency is {asap_latency}"
            )

        node_by_id = {node.id: node for node in region.nodes}
        outgoing: dict[str, list[str]] = {node.id: [] for node in region.nodes}

        for edge in self.get_data_edges(region):
            outgoing[edge.source].append(edge.target)
        topo_order = topological_sort(
            region.nodes,
            lambda current: (node_by_id[succ_id] for succ_id in outgoing[current.id]),
            key=lambda current: current.id,
            cycle_error=lambda _: ValueError(f"region '{region.id}' is cyclic and cannot be ALAP-scheduled"),
        )

        latest_starts: dict[str, int] = {}
        for node in reversed(topo_order):
            node_id = node.id
            successors = outgoing[node_id]
            if not successors:
                start = max_latency if node_id == sink.id else max_latency - _occupied_steps(node)
            else:
                start = min(latest_starts[succ_id] - _completion_delay(node) for succ_id in successors)
            if start < 0:
                raise ValueError(
                    f"ALAPScheduler cannot schedule region '{region.id}' with latency {max_latency}; "
                    f"node '{node_id}' would start at {start}"
                )
            latest_starts[node_id] = start

        starts = {node_id: (start, _scheduled_end(node_by_id[node_id], start)) for node_id, start in latest_starts.items()}
        return SGUScheduleResult(region.id, starts, latency=max_latency)


def _resolve_region_latency_max(region: UHIRRegion, spec: dict[str, object]) -> int:
    mode = spec.get("mode")
    if mode == "asap":
        slack = spec.get("slack", 0)
        if not isinstance(slack, int) or slack < 0:
            raise ValueError("ALAPScheduler asap latency mode requires non-negative integer slack")
        return ASAPScheduler().schedule_sgu(region).latency + slack
    if mode == "explicit":
        values = spec.get("values")
        if not isinstance(values, dict) or region.id not in values:
            raise ValueError(f"ALAPScheduler requires one latency max for region '{region.id}'")
        value = values[region.id]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"ALAPScheduler requires non-negative integer latency max for region '{region.id}'")
        return value
    raise ValueError("ALAPScheduler received unsupported sgu_latency_max mode")


def _completion_delay(node: UHIRNode) -> int:
    delay = node.attributes.get("delay")
    if not isinstance(delay, int) or delay < 0:
        raise ValueError(f"node '{node.id}' must declare one non-negative integer delay")
    return delay


def _occupied_steps(node: UHIRNode) -> int:
    delay = _completion_delay(node)
    if delay == 0:
        return 1
    return delay


def _scheduled_end(node: UHIRNode, start: int) -> int:
    delay = _completion_delay(node)
    if delay == 0:
        return start
    return start + delay - 1
