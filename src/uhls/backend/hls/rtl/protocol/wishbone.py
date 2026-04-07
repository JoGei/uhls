"""Generic Wishbone protocol planning shared across RTL backends."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re

from uhls.backend.hls.uhir.model import UHIRAssign, UHIRConstant, UHIRDesign, UHIRPort, UHIRResource, UHIRSeqBlock, UHIRSeqUpdate

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


def build_wishbone_slave_wrapper_uglir(core_design: UHIRDesign, wrapper: SlaveWrapperPlan, protocol: WishboneSlaveProtocolPlan) -> UHIRDesign:
    """Merge a Wishbone slave wrapper around one core µglIR design."""
    design = deepcopy(core_design)
    if design.stage != "uglir":
        raise ValueError(f"wishbone wrapper builder expects µglIR input, got stage '{design.stage}'")

    original_inputs = list(design.inputs)
    original_outputs = list(design.outputs)
    core_scalar_input_names = {port.name for port in wrapper.scalar_inputs}
    core_scalar_output_names = {port.name for port in wrapper.scalar_outputs}
    memory_bases = {interface.base for interface in wrapper.memory_interfaces}

    design.inputs = [UHIRPort(port.direction, port.name, port.type) for port in protocol.ports if port.direction == "input"]
    design.outputs = [UHIRPort(port.direction, port.name, port.type) for port in protocol.ports if port.direction == "output"]

    for port in original_inputs:
        if port.name not in {"clk", "rst"}:
            design.resources.append(UHIRResource("net", port.name, port.type))
    for port in original_outputs:
        design.resources.append(UHIRResource("net", port.name, port.type))

    design.constants.append(UHIRConstant("WB_REG_CONTROL_STATUS", f"WB_BASE_ADDR + {_u32(protocol.control_status_address)}", "u32"))
    for port in protocol.scalar_inputs:
        design.constants.append(UHIRConstant(port.symbol, f"WB_BASE_ADDR + {_u32(port.address)}", "u32"))
    for port in protocol.scalar_outputs:
        design.constants.append(UHIRConstant(port.symbol, f"WB_BASE_ADDR + {_u32(port.address)}", "u32"))
    for window in protocol.memory_windows:
        design.constants.append(UHIRConstant(window.symbol, f"WB_BASE_ADDR + {_u32(window.base_address)}", "u32"))

    design.resources.extend(
        [
            UHIRResource("net", "wb_req_n", "i1"),
            UHIRResource("reg", "start_pending_q", "i1"),
            UHIRResource("reg", "busy_q", "i1"),
            UHIRResource("reg", "done_q", "i1"),
        ]
    )
    for port in wrapper.scalar_inputs:
        design.resources.append(UHIRResource("reg", f"{port.name}_q", port.type))
    for port in wrapper.scalar_outputs:
        design.resources.append(UHIRResource("reg", f"{port.name}_q", port.type))
    for interface in wrapper.memory_interfaces:
        depth = _memory_depth(interface)
        design.resources.append(UHIRResource("mem", f"{interface.base}_mem_q", f"{interface.data_type}[{depth}]"))
        design.resources.append(UHIRResource("net", f"{interface.base}_bus_hit_n", "i1"))
        design.resources.append(UHIRResource("net", f"{interface.base}_bus_word_addr_n", f"u{_memory_index_width(interface)}"))

    design.assigns.extend(
        [
            UHIRAssign("wb_req_n", "wb_cyc_i & wb_stb_i"),
            UHIRAssign("wb_ack_o", "wb_req_n"),
            UHIRAssign("req_valid", "start_pending_q"),
            UHIRAssign("resp_ready", "true"),
            UHIRAssign("wb_dat_o", _wishbone_read_data_expr(wrapper, protocol)),
        ]
    )
    for port in wrapper.scalar_inputs:
        design.assigns.append(UHIRAssign(port.name, f"{port.name}_q"))
    for interface in wrapper.memory_interfaces:
        window = next(window for window in protocol.memory_windows if window.base == interface.base)
        design.assigns.append(
            UHIRAssign(
                f"{interface.base}_bus_hit_n",
                f"wb_adr_i >= {window.symbol} && wb_adr_i < ({window.symbol} + {_u32(window.span_bytes)})",
            )
        )
        design.assigns.append(UHIRAssign(f"{interface.base}_bus_word_addr_n", "wb_adr_i[11:2]"))
        design.assigns.append(
            UHIRAssign(
                f"{interface.base}_rdata",
                f"{interface.base}_mem_q[{_memory_index_expr(f'{interface.base}_addr', interface)}]",
            )
        )

    seq = UHIRSeqBlock(clock="clk", reset="rst")
    seq.reset_updates.extend(
        [
            UHIRSeqUpdate("start_pending_q", "false"),
            UHIRSeqUpdate("busy_q", "false"),
            UHIRSeqUpdate("done_q", "false"),
        ]
    )
    for port in wrapper.scalar_inputs:
        seq.reset_updates.append(UHIRSeqUpdate(f"{port.name}_q", _zero_expr_for_type(port.type)))
    for port in wrapper.scalar_outputs:
        seq.reset_updates.append(UHIRSeqUpdate(f"{port.name}_q", _zero_expr_for_type(port.type)))

    control_write_cond = "(wb_req_n && wb_we_i) && (wb_adr_i == WB_REG_CONTROL_STATUS) && wb_dat_i[0]"
    seq.updates.append(
        UHIRSeqUpdate(
            "start_pending_q",
            f"({control_write_cond}) ? true : ((start_pending_q && req_ready) ? false : start_pending_q)",
        )
    )
    seq.updates.append(
        UHIRSeqUpdate(
            "busy_q",
            "((start_pending_q && req_ready) ? true : (resp_valid ? false : busy_q))",
        )
    )
    seq.updates.append(
        UHIRSeqUpdate(
            "done_q",
            f"({control_write_cond}) ? false : (resp_valid ? true : done_q)",
        )
    )
    for port in protocol.scalar_inputs:
        seq.updates.append(
            UHIRSeqUpdate(
                f"{port.name}_q",
                f"((wb_req_n && wb_we_i) && (wb_adr_i == {port.symbol})) ? {_unpack_from_wishbone('wb_dat_i', port.type)} : {port.name}_q",
            )
        )
    for port in wrapper.scalar_outputs:
        seq.updates.append(
            UHIRSeqUpdate(
                f"{port.name}_q",
                f"resp_valid ? {port.name} : {port.name}_q",
            )
        )
    for interface in wrapper.memory_interfaces:
        if interface.has_write:
            seq.updates.append(
                UHIRSeqUpdate(
                    f"{interface.base}_mem_q[{interface.base}_bus_word_addr_n]",
                    _unpack_from_wishbone("wb_dat_i", interface.data_type),
                    f"(wb_req_n && wb_we_i) && {interface.base}_bus_hit_n",
                )
            )
            seq.updates.append(
                UHIRSeqUpdate(
                    f"{interface.base}_mem_q[{_memory_index_expr(f'{interface.base}_addr', interface)}]",
                    f"{interface.base}_wdata",
                    f"{interface.base}_we",
                )
            )
    design.seq_blocks.append(seq)

    _dedupe_resources(design)
    _dedupe_constants(design)
    _drop_obsolete_interface_ports(design, original_inputs, original_outputs, core_scalar_input_names, core_scalar_output_names, memory_bases)
    return design


def _drop_obsolete_interface_ports(
    design: UHIRDesign,
    original_inputs: list[UHIRPort],
    original_outputs: list[UHIRPort],
    scalar_input_names: set[str],
    scalar_output_names: set[str],
    memory_bases: set[str],
) -> None:
    current_input_names = {port.name for port in design.inputs}
    current_output_names = {port.name for port in design.outputs}
    for port in original_inputs:
        if port.name in {"clk", "rst"} or port.name in current_input_names:
            continue
        if port.name in scalar_input_names or any(port.name == f"{base}_rdata" for base in memory_bases):
            continue
    for port in original_outputs:
        if port.name in current_output_names:
            continue
        if port.name in scalar_output_names or any(port.name in {f"{base}_addr", f"{base}_wdata", f"{base}_we"} for base in memory_bases):
            continue


def _dedupe_resources(design: UHIRDesign) -> None:
    deduped: list[UHIRResource] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for resource in design.resources:
        key = (resource.kind, resource.id, resource.value, resource.target)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resource)
    design.resources = deduped


def _dedupe_constants(design: UHIRDesign) -> None:
    deduped: list[UHIRConstant] = []
    seen: set[tuple[str, int | str, str]] = set()
    for constant in design.constants:
        key = (constant.name, constant.value, constant.type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(constant)
    design.constants = deduped


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


def _u32(value: int) -> str:
    upper = (value >> 16) & 0xFFFF
    lower = value & 0xFFFF
    return f"32'h{upper:04x}_{lower:04x}"


def _zero_expr_for_type(type_hint: str) -> str:
    match = re.fullmatch(r"([iu])(\d+)", type_hint)
    if match is None:
        return "0:u1"
    return f"0:{match.group(1)}{match.group(2)}"


def _pack_to_wishbone(signal: str, type_hint: str) -> str:
    match = re.fullmatch(r"([iu])(\d+)", type_hint)
    if match is None:
        return signal
    signed = match.group(1) == "i"
    width = int(match.group(2))
    if width >= 32:
        return signal if width == 32 else f"{signal}[31:0]"
    if signed:
        return f"{{{{{32 - width}{{{signal}[{width - 1}]}}}}, {signal}}}"
    return f"{{0:u{32 - width}, {signal}}}"


def _unpack_from_wishbone(signal: str, type_hint: str) -> str:
    match = re.fullmatch(r"([iu])(\d+)", type_hint)
    if match is None:
        return signal
    width = int(match.group(2))
    if width >= 32:
        return signal if width == 32 else f"{signal}[{width - 1}:0]"
    if width == 1:
        return f"{signal}[0]"
    return f"{signal}[{width - 1}:0]"


def _memory_depth(interface) -> int:
    depth = getattr(interface, "depth", None)
    if isinstance(depth, int) and depth > 0:
        return depth
    return 1024


def _memory_index_width(interface) -> int:
    depth = _memory_depth(interface)
    return max(depth - 1, 1).bit_length()


def _memory_index_expr(address_expr: str, interface) -> str:
    width = _memory_index_width(interface)
    if width <= 1:
        return f"{address_expr}[0]"
    return f"{address_expr}[{width - 1}:0]"


def _wishbone_read_data_expr(wrapper: SlaveWrapperPlan, protocol: WishboneSlaveProtocolPlan) -> str:
    read_expr = "0:u32"
    for interface in reversed(wrapper.memory_interfaces):
        read_expr = (
            f"{interface.base}_bus_hit_n ? "
            f"{_pack_to_wishbone(f'{interface.base}_mem_q[{interface.base}_bus_word_addr_n]', interface.data_type)} : "
            f"({read_expr})"
        )
    for port in reversed(protocol.scalar_outputs):
        read_expr = f"(wb_adr_i == {port.symbol}) ? {_pack_to_wishbone(port.name + '_q', port.type)} : ({read_expr})"
    for port in reversed(protocol.scalar_inputs):
        read_expr = f"(wb_adr_i == {port.symbol}) ? {_pack_to_wishbone(port.name + '_q', port.type)} : ({read_expr})"
    status_expr = "{28:u28, req_ready, start_pending_q, busy_q, done_q}"
    read_expr = f"(wb_adr_i == WB_REG_CONTROL_STATUS) ? {status_expr} : ({read_expr})"
    return f"((wb_req_n && !wb_we_i) ? ({read_expr}) : 0:u32)"
