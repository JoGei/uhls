"""Data model for textual µhIR artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field

AttributeValue = str | int | bool | tuple[str, ...]


@dataclass(slots=True, frozen=True)
class UHIRPort:
    """One top-level interface declaration."""

    direction: str
    name: str
    type: str


@dataclass(slots=True, frozen=True)
class UHIRConstant:
    """One top-level constant declaration."""

    name: str
    value: int | str
    type: str


@dataclass(slots=True, frozen=True)
class UHIRSchedule:
    """One schedule declaration."""

    kind: str


@dataclass(slots=True)
class UHIRNode:
    """One region-local operation node."""

    id: str
    opcode: str
    operands: tuple[str, ...] = ()
    result_type: str | None = None
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass(slots=True)
class UHIREdge:
    """One dependency or sequencing edge."""

    kind: str
    source: str
    target: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    directed: bool = True


@dataclass(slots=True, frozen=True)
class UHIRSourceMap:
    """One provenance mapping from a µhIR node to an earlier source id."""

    node_id: str
    source_id: str


@dataclass(slots=True, frozen=True)
class UHIRRegionRef:
    """One hierarchical region reference inside another region."""

    target: str


@dataclass(slots=True, frozen=True)
class UHIRValueBinding:
    """One bind-stage value-to-register binding."""

    producer: str
    register: str
    live_start: int
    live_end: int


@dataclass(slots=True)
class UHIRMux:
    """One bind-stage mux declaration."""

    id: str
    inputs: tuple[str, ...]
    output: str
    select: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UHIRResource:
    """One bind-stage resource declaration."""

    kind: str
    id: str
    value: str
    target: str | None = None


@dataclass(slots=True)
class UHIRRegion:
    """One µhIR region block."""

    id: str
    kind: str
    parent: str | None = None
    region_refs: list[UHIRRegionRef] = field(default_factory=list)
    nodes: list[UHIRNode] = field(default_factory=list)
    edges: list[UHIREdge] = field(default_factory=list)
    mappings: list[UHIRSourceMap] = field(default_factory=list)
    value_bindings: list[UHIRValueBinding] = field(default_factory=list)
    muxes: list[UHIRMux] = field(default_factory=list)
    steps: tuple[int, int] | None = None
    latency: int | None = None
    initiation_interval: int | None = None


@dataclass(slots=True)
class UHIRDesign:
    """One parsed µhIR design artifact."""

    name: str
    stage: str
    inputs: list[UHIRPort] = field(default_factory=list)
    outputs: list[UHIRPort] = field(default_factory=list)
    constants: list[UHIRConstant] = field(default_factory=list)
    schedule: UHIRSchedule | None = None
    resources: list[UHIRResource] = field(default_factory=list)
    regions: list[UHIRRegion] = field(default_factory=list)

    def get_region(self, region_id: str) -> UHIRRegion | None:
        """Return one region by id."""
        for region in self.regions:
            if region.id == region_id:
                return region
        return None
