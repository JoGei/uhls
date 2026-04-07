"""Generic Wishbone protocol planning shared across RTL backends."""

from __future__ import annotations

from dataclasses import dataclass

from ..wrap import SlaveWrapperPlan


@dataclass(frozen=True)
class ProtocolPort:
    """One protocol port independent of the emitted HDL syntax."""

    direction: str
    type: str
    name: str


@dataclass(frozen=True)
class WishboneScalarRegister:
    """One scalar register mapped into one Wishbone address."""

    name: str
    type: str
    address: int
    symbol: str


@dataclass(frozen=True)
class WishboneMemoryWindow:
    """One memory window mapped into one Wishbone address range."""

    base: str
    data_type: str
    addr_type: str | None
    write_type: str | None
    has_write: bool
    depth: int | None
    base_address: int
    span_bytes: int
    symbol: str


@dataclass(frozen=True)
class WishboneSlaveProtocolPlan:
    """One generic Wishbone-slave protocol plan."""

    ports: tuple[ProtocolPort, ...]
    control_status_address: int
    scalar_inputs: tuple[WishboneScalarRegister, ...]
    scalar_outputs: tuple[WishboneScalarRegister, ...]
    memory_windows: tuple[WishboneMemoryWindow, ...]


def plan_wishbone_slave_protocol(wrapper: SlaveWrapperPlan) -> WishboneSlaveProtocolPlan:
    """Build the generic Wishbone-slave protocol plan for one slave wrapper."""
    scalar_inputs = tuple(
        WishboneScalarRegister(
            name=port.name,
            type=port.type,
            address=0x0100 + 4 * index,
            symbol=f"WB_REG_IN_{_sanitize_symbol(port.name)}",
        )
        for index, port in enumerate(wrapper.scalar_inputs)
    )
    scalar_outputs = tuple(
        WishboneScalarRegister(
            name=port.name,
            type=port.type,
            address=0x0200 + 4 * index,
            symbol=f"WB_REG_OUT_{_sanitize_symbol(port.name)}",
        )
        for index, port in enumerate(wrapper.scalar_outputs)
    )
    memory_windows = tuple(
        WishboneMemoryWindow(
            base=interface.base,
            data_type=interface.data_type,
            addr_type=interface.addr_type,
            write_type=interface.write_type,
            has_write=interface.has_write,
            depth=interface.depth,
            base_address=0x1000 + 0x1000 * index,
            span_bytes=_memory_window_span_bytes(interface.data_type, interface.depth),
            symbol=f"WB_MEM_{_sanitize_symbol(interface.base)}_BASE",
        )
        for index, interface in enumerate(wrapper.memory_interfaces)
    )
    return WishboneSlaveProtocolPlan(
        ports=(
            ProtocolPort("input", "clock", "clk"),
            ProtocolPort("input", "i1", "rst"),
            ProtocolPort("input", "i1", "wb_cyc_i"),
            ProtocolPort("input", "i1", "wb_stb_i"),
            ProtocolPort("input", "i1", "wb_we_i"),
            ProtocolPort("input", "u32", "wb_adr_i"),
            ProtocolPort("input", "u32", "wb_dat_i"),
            ProtocolPort("input", "u4", "wb_sel_i"),
            ProtocolPort("output", "u32", "wb_dat_o"),
            ProtocolPort("output", "i1", "wb_ack_o"),
        ),
        control_status_address=0x0000,
        scalar_inputs=scalar_inputs,
        scalar_outputs=scalar_outputs,
        memory_windows=memory_windows,
    )


def _sanitize_symbol(text: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")
    if not sanitized:
        sanitized = "VALUE"
    if sanitized[0].isdigit():
        sanitized = f"V_{sanitized}"
    return sanitized.upper()


def _memory_window_span_bytes(type_name: str, depth: int | None) -> int:
    if depth is None:
        return 0x1000
    return max(_type_size_bytes(type_name) * depth, _type_size_bytes(type_name))


def _type_size_bytes(type_name: str) -> int:
    if len(type_name) >= 2 and type_name[0] in {"i", "u"} and type_name[1:].isdigit():
        width = int(type_name[1:])
        return max((width + 7) // 8, 1)
    return 4
