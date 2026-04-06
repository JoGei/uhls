"""Interfaces for pluggable SGU schedulers."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Callable, Protocol

from uhls.backend.uhir.model import TimingValue, UHIREdge, UHIRRegion


ScheduleInterval = tuple[TimingValue, TimingValue]


@dataclass(slots=True, frozen=True)
class SGUScheduleResult:
    """One scheduler result for one flat sequencing graph unit."""

    region_id: str
    node_starts: dict[str, ScheduleInterval]
    latency: TimingValue
    initiation_interval: TimingValue | None = None
    steps: ScheduleInterval | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class SGUScheduler(Protocol):
    """Interface implemented by one user-supplied flat SGU scheduler."""

    def schedule_sgu(self, region: UHIRRegion) -> SGUScheduleResult:
        """Return one local schedule for one allocated SGU region."""


class SGUSchedulerBase(ABC):
    """Shared validation helpers for flat SGU schedulers."""

    def get_source(self, region: UHIRRegion):
        """Return one region's unique source nop."""
        sources = [node for node in region.nodes if node.opcode == "nop" and node.attributes.get("role") == "source"]
        if len(sources) != 1:
            names = ", ".join(node.id for node in sources) or "<none>"
            raise ValueError(f"region '{region.id}' must contain exactly one source nop; found: {names}")
        return sources[0]

    def get_sink(self, region: UHIRRegion):
        """Return one region's unique sink nop."""
        sinks = [node for node in region.nodes if node.opcode == "nop" and node.attributes.get("role") == "sink"]
        if len(sinks) != 1:
            names = ", ".join(node.id for node in sinks) or "<none>"
            raise ValueError(f"region '{region.id}' must contain exactly one sink nop; found: {names}")
        return sinks[0]

    def get_data_edges(self, region: UHIRRegion) -> Iterator[UHIREdge]:
        """Yield one region's flat intra-SGU dependency edges."""
        node_ids = {node.id for node in region.nodes}
        for edge in region.edges:
            if edge.kind == "seq":
                continue
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(
                    f"region '{region.id}' contains non-local data edge '{edge.source} -> {edge.target}' of kind '{edge.kind}'"
                )
            yield edge

    def get_seq_edges(self, region: UHIRRegion) -> Iterator[UHIREdge]:
        """Yield one region's hierarchical sequencing edges."""
        for edge in region.edges:
            if edge.kind == "seq":
                yield edge

    def validate_sgu_region(self, region: UHIRRegion) -> None:
        """Assert one canonical flat SGU shape before scheduling it."""
        node_by_id = {node.id: node for node in region.nodes}
        outgoing_counts: dict[str, int] = {node.id: 0 for node in region.nodes}
        self.get_source(region)

        for edge in self.get_data_edges(region):
            outgoing_counts[edge.source] += 1

        leaves = [node_by_id[node_id] for node_id, count in outgoing_counts.items() if count == 0]
        if len(leaves) != 1:
            names = ", ".join(node.id for node in leaves) or "<none>"
            raise ValueError(
                f"region '{region.id}' must contain exactly one leaf node, and it must be the sink nop; found: {names}"
            )

        sink = leaves[0]
        if sink.id != self.get_sink(region).id:
            raise ValueError(
                f"region '{region.id}' leaf node '{sink.id}' must be the sink nop, got opcode='{sink.opcode}' role='{sink.attributes.get('role')}'"
            )


@dataclass(slots=True, frozen=True)
class CallableSGUScheduler:
    """Adapter that exposes one plain callable as one SGU scheduler."""

    callback: Callable[[UHIRRegion], SGUScheduleResult]

    def schedule_sgu(self, region: UHIRRegion) -> SGUScheduleResult:
        return self.callback(region)
