"""Data model for textual µhIR artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .timing import TimingExpr

AttributeValue = str | int | bool | tuple[str, ...] | TimingExpr
TimingValue = int | TimingExpr


@dataclass(slots=True, frozen=True)
class UHIRPort:
    """One top-level interface declaration."""

    direction: str
    name: str
    type: str


@dataclass(slots=True)
class UHIRController:
    """One fsm-stage controller declaration."""

    name: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    inputs: list[UHIRPort] = field(default_factory=list)
    outputs: list[UHIRPort] = field(default_factory=list)
    states: list["UHIRControllerState"] = field(default_factory=list)
    transitions: list["UHIRControllerTransition"] = field(default_factory=list)
    emits: list["UHIRControllerEmit"] = field(default_factory=list)
    links: list["UHIRControllerLink"] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class UHIRControllerState:
    """One fsm-stage controller state declaration."""

    name: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UHIRControllerTransition:
    """One fsm-stage controller transition declaration."""

    source: str
    target: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UHIRControllerEmit:
    """One fsm-stage controller per-state output/action bundle."""

    state: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UHIRControllerLink:
    """One explicit handshake link between two controllers."""

    child: str
    node: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UHIRAssign:
    """One uglir combinational assignment."""

    target: str
    expr: str


@dataclass(slots=True, frozen=True)
class UHIRAttach:
    """One uglir instance-port attachment."""

    instance: str
    port: str
    signal: str


@dataclass(slots=True, frozen=True)
class UHIRGlueMuxCase:
    """One uglir mux alternative."""

    key: str
    source: str


@dataclass(slots=True)
class UHIRGlueMux:
    """One uglir explicit glue mux declaration."""

    name: str
    type: str
    select: str
    cases: list[UHIRGlueMuxCase] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class UHIRSeqUpdate:
    """One uglir sequential register update."""

    target: str
    value: str
    enable: str | None = None


@dataclass(slots=True)
class UHIRSeqBlock:
    """One uglir sequential block with one optional reset branch."""

    clock: str
    reset: str | None = None
    reset_updates: list[UHIRSeqUpdate] = field(default_factory=list)
    updates: list[UHIRSeqUpdate] = field(default_factory=list)


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
    """One bind-stage source-value-to-register binding."""

    producer: str
    register: str
    live_intervals: tuple[tuple[int, int], ...]

    @property
    def live_start(self) -> int:
        """Compatibility accessor for one binding's first live step."""
        return self.live_intervals[0][0]

    @property
    def live_end(self) -> int:
        """Compatibility accessor for one binding's last live step."""
        return self.live_intervals[-1][1]


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
    steps: tuple[TimingValue, TimingValue] | None = None
    latency: TimingValue | None = None
    initiation_interval: TimingValue | None = None


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
    controllers: list[UHIRController] = field(default_factory=list)
    assigns: list[UHIRAssign] = field(default_factory=list)
    attachments: list[UHIRAttach] = field(default_factory=list)
    glue_muxes: list[UHIRGlueMux] = field(default_factory=list)
    seq_blocks: list[UHIRSeqBlock] = field(default_factory=list)
    regions: list[UHIRRegion] = field(default_factory=list)

    def get_region(self, region_id: str) -> UHIRRegion | None:
        """Return one region by id."""
        for region in self.regions:
            if region.id == region_id:
                return region
        return None
