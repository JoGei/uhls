"""Data model for textual µglIR artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class UGLIRPort:
    """One top-level µglIR interface declaration."""

    direction: str
    name: str
    type: str


@dataclass(slots=True, frozen=True)
class UGLIRConstant:
    """One top-level µglIR constant declaration."""

    name: str
    value: int | str
    type: str


@dataclass(slots=True, frozen=True)
class UGLIRAddressMapEntry:
    """One software-visible address-map entry."""

    kind: str
    name: str
    attributes: dict[str, str | int | bool | tuple[str, ...]] = field(default_factory=dict)


@dataclass(slots=True)
class UGLIRAddressMap:
    """One top-level address map declaration."""

    name: str
    entries: list[UGLIRAddressMapEntry] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class UGLIRResource:
    """One µglIR resource declaration."""

    kind: str
    id: str
    value: str
    target: str | None = None


@dataclass(slots=True, frozen=True)
class UGLIRAssign:
    """One combinational assignment."""

    target: str
    expr: str


@dataclass(slots=True, frozen=True)
class UGLIRAttach:
    """One instance-port attachment."""

    instance: str
    port: str
    signal: str


@dataclass(slots=True, frozen=True)
class UGLIRMuxCase:
    """One explicit mux alternative."""

    key: str
    source: str


@dataclass(slots=True)
class UGLIRMux:
    """One explicit mux declaration."""

    name: str
    type: str
    select: str
    cases: list[UGLIRMuxCase] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class UGLIRSeqUpdate:
    """One sequential update."""

    target: str
    value: str
    enable: str | None = None


@dataclass(slots=True)
class UGLIRSeqBlock:
    """One sequential block with one optional reset branch."""

    clock: str
    reset: str | None = None
    reset_updates: list[UGLIRSeqUpdate] = field(default_factory=list)
    updates: list[UGLIRSeqUpdate] = field(default_factory=list)


@dataclass(slots=True)
class UGLIRDesign:
    """One parsed µglIR design artifact."""

    name: str
    stage: str = "uglir"
    inputs: list[UGLIRPort] = field(default_factory=list)
    outputs: list[UGLIRPort] = field(default_factory=list)
    constants: list[UGLIRConstant] = field(default_factory=list)
    address_maps: list[UGLIRAddressMap] = field(default_factory=list)
    resources: list[UGLIRResource] = field(default_factory=list)
    assigns: list[UGLIRAssign] = field(default_factory=list)
    attachments: list[UGLIRAttach] = field(default_factory=list)
    muxes: list[UGLIRMux] = field(default_factory=list)
    seq_blocks: list[UGLIRSeqBlock] = field(default_factory=list)


def to_uglir_design(design) -> UGLIRDesign:
    """Normalize one design-like object into one concrete UGLIRDesign."""
    if isinstance(design, UGLIRDesign):
        return design
    return UGLIRDesign(
        name=design.name,
        stage=design.stage,
        inputs=[UGLIRPort(port.direction, port.name, port.type) for port in design.inputs],
        outputs=[UGLIRPort(port.direction, port.name, port.type) for port in design.outputs],
        constants=[UGLIRConstant(const.name, const.value, const.type) for const in design.constants],
        address_maps=[
            UGLIRAddressMap(
                address_map.name,
                [UGLIRAddressMapEntry(entry.kind, entry.name, dict(entry.attributes)) for entry in address_map.entries],
            )
            for address_map in design.address_maps
        ],
        resources=[UGLIRResource(resource.kind, resource.id, resource.value, resource.target) for resource in design.resources],
        assigns=[UGLIRAssign(assign.target, assign.expr) for assign in design.assigns],
        attachments=[UGLIRAttach(attachment.instance, attachment.port, attachment.signal) for attachment in design.attachments],
        muxes=[
            UGLIRMux(
                mux.name,
                mux.type,
                mux.select,
                [UGLIRMuxCase(case.key, case.source) for case in mux.cases],
            )
            for mux in design.muxes
        ],
        seq_blocks=[
            UGLIRSeqBlock(
                seq_block.clock,
                seq_block.reset,
                [UGLIRSeqUpdate(update.target, update.value, update.enable) for update in seq_block.reset_updates],
                [UGLIRSeqUpdate(update.target, update.value, update.enable) for update in seq_block.updates],
            )
            for seq_block in design.seq_blocks
        ],
    )


__all__ = [
    "UGLIRAddressMap",
    "UGLIRAddressMapEntry",
    "UGLIRAssign",
    "UGLIRAttach",
    "UGLIRConstant",
    "UGLIRDesign",
    "UGLIRMux",
    "UGLIRMuxCase",
    "UGLIRPort",
    "UGLIRResource",
    "UGLIRSeqBlock",
    "UGLIRSeqUpdate",
    "to_uglir_design",
]
