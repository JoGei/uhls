"""Generic OBI protocol planning shared across RTL backends."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re

from uhls.backend.hls.uglir.model import (
    UGLIRAddressMap,
    UGLIRAddressMapEntry,
    UGLIRAssign,
    UGLIRConstant,
    UGLIRDesign,
    UGLIRPort,
    UGLIRResource,
    UGLIRSeqBlock,
    UGLIRSeqUpdate,
)

from ..wrap import SlaveWrapperPlan
from .wishbone import ProtocolPort


@dataclass(frozen=True)
class OBIScalarRegister:
    """One scalar register mapped into one OBI address."""

    name: str
    type: str
    address: int
    symbol: str


@dataclass(frozen=True)
class OBIMemoryWindow:
    """One memory window mapped into one OBI address range."""

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
class OBISlaveProtocolPlan:
    """One generic OBI-subordinate protocol plan."""

    ports: tuple[ProtocolPort, ...]
    control_status_address: int
    scalar_inputs: tuple[OBIScalarRegister, ...]
    scalar_outputs: tuple[OBIScalarRegister, ...]
    memory_windows: tuple[OBIMemoryWindow, ...]
    features: tuple[str, ...] = ()
    burst_capable: bool = False
    response_queue_depth: int = 1


def plan_obi_slave_protocol(wrapper: SlaveWrapperPlan, features: tuple[str, ...] = ()) -> OBISlaveProtocolPlan:
    """Build the generic OBI-subordinate protocol plan for one slave wrapper."""
    burst_capable = "burst" in features
    response_queue_depth = 2 if burst_capable else 1
    scalar_inputs = tuple(
        OBIScalarRegister(
            name=port.name,
            type=port.type,
            address=0x0100 + 4 * index,
            symbol=f"OBI_REG_IN_{_sanitize_symbol(port.name)}",
        )
        for index, port in enumerate(wrapper.scalar_inputs)
    )
    scalar_outputs = tuple(
        OBIScalarRegister(
            name=port.name,
            type=port.type,
            address=0x0200 + 4 * index,
            symbol=f"OBI_REG_OUT_{_sanitize_symbol(port.name)}",
        )
        for index, port in enumerate(wrapper.scalar_outputs)
    )
    memory_windows = tuple(
        OBIMemoryWindow(
            base=interface.base,
            data_type=interface.data_type,
            addr_type=interface.addr_type,
            write_type=interface.write_type,
            has_write=interface.has_write,
            depth=interface.depth,
            base_address=0x1000 + 0x1000 * index,
            span_bytes=_memory_window_span_bytes(interface.data_type, interface.depth),
            symbol=f"OBI_MEM_{_sanitize_symbol(interface.base)}_BASE",
        )
        for index, interface in enumerate(wrapper.memory_interfaces)
    )
    ports = (
        ProtocolPort("input", "clock", "clk"),
        ProtocolPort("input", "i1", "rst"),
        ProtocolPort("input", "i1", "obi_req_i"),
        ProtocolPort("input", "u32", "obi_addr_i"),
        ProtocolPort("input", "i1", "obi_we_i"),
        ProtocolPort("input", "u4", "obi_be_i"),
        ProtocolPort("input", "u32", "obi_wdata_i"),
        ProtocolPort("input", "i1", "obi_rready_i"),
        ProtocolPort("output", "i1", "obi_gnt_o"),
        ProtocolPort("output", "i1", "obi_rvalid_o"),
        ProtocolPort("output", "u32", "obi_rdata_o"),
    )
    return OBISlaveProtocolPlan(
        ports=ports,
        control_status_address=0x0000,
        scalar_inputs=scalar_inputs,
        scalar_outputs=scalar_outputs,
        memory_windows=memory_windows,
        features=features,
        burst_capable=burst_capable,
        response_queue_depth=response_queue_depth,
    )


def build_obi_slave_wrapper_uglir(
    core_design: UGLIRDesign,
    wrapper: SlaveWrapperPlan,
    protocol: OBISlaveProtocolPlan,
) -> UGLIRDesign:
    """Merge one OBI subordinate wrapper around one core µglIR design."""
    design = deepcopy(core_design)
    if design.stage != "uglir":
        raise ValueError(f"obi wrapper builder expects µglIR input, got stage '{design.stage}'")

    original_inputs = list(design.inputs)
    original_outputs = list(design.outputs)
    core_scalar_input_names = {port.name for port in wrapper.scalar_inputs}
    core_scalar_output_names = {port.name for port in wrapper.scalar_outputs}
    memory_bases = {interface.base for interface in wrapper.memory_interfaces}

    design.inputs = [UGLIRPort(port.direction, port.name, port.type) for port in protocol.ports if port.direction == "input"]
    design.outputs = [UGLIRPort(port.direction, port.name, port.type) for port in protocol.ports if port.direction == "output"]

    for port in original_inputs:
        if port.name not in {"clk", "rst"}:
            design.resources.append(UGLIRResource("net", port.name, port.type))
    for port in original_outputs:
        design.resources.append(UGLIRResource("net", port.name, port.type))

    design.constants.append(UGLIRConstant("OBI_REG_CONTROL_STATUS", f"OBI_BASE_ADDR + {_u32(protocol.control_status_address)}", "u32"))
    for port in protocol.scalar_inputs:
        design.constants.append(UGLIRConstant(port.symbol, f"OBI_BASE_ADDR + {_u32(port.address)}", "u32"))
    for port in protocol.scalar_outputs:
        design.constants.append(UGLIRConstant(port.symbol, f"OBI_BASE_ADDR + {_u32(port.address)}", "u32"))
    for window in protocol.memory_windows:
        design.constants.append(UGLIRConstant(window.symbol, f"OBI_BASE_ADDR + {_u32(window.base_address)}", "u32"))
    design.address_maps.append(_obi_address_map(protocol))

    design.resources.extend(
        [
            UGLIRResource("net", "obi_accept_n", "i1"),
            UGLIRResource("net", "obi_hit_n", "i1"),
            UGLIRResource("reg", "start_pending_q", "i1"),
            UGLIRResource("reg", "busy_q", "i1"),
            UGLIRResource("reg", "done_q", "i1"),
        ]
    )
    if protocol.response_queue_depth <= 1:
        design.resources.extend(
            [
                UGLIRResource("reg", "obi_rsp_pending_q", "i1"),
                UGLIRResource("reg", "obi_rsp_rdata_q", "u32"),
            ]
        )
    else:
        depth = protocol.response_queue_depth
        index_width = _queue_index_width(depth)
        count_width = _queue_count_width(depth)
        design.resources.extend(
            [
                UGLIRResource("mem", "obi_rsp_fifo_q", f"u32[{depth}]"),
                UGLIRResource("reg", "obi_rsp_head_q", f"u{index_width}"),
                UGLIRResource("reg", "obi_rsp_tail_q", f"u{index_width}"),
                UGLIRResource("reg", "obi_rsp_count_q", f"u{count_width}"),
            ]
        )
    for port in wrapper.scalar_inputs:
        design.resources.append(UGLIRResource("reg", f"{port.name}_q", port.type))
    for port in wrapper.scalar_outputs:
        design.resources.append(UGLIRResource("reg", f"{port.name}_q", port.type))
    for interface in wrapper.memory_interfaces:
        depth = _memory_depth(interface)
        design.resources.append(UGLIRResource("mem", f"{interface.base}_mem_q", f"{interface.data_type}[{depth}]"))
        design.resources.append(UGLIRResource("net", f"{interface.base}_bus_hit_n", "i1"))
        design.resources.append(UGLIRResource("net", f"{interface.base}_bus_word_addr_n", f"u{_memory_index_width(interface)}"))

    if protocol.response_queue_depth <= 1:
        design.assigns.extend(
            [
                UGLIRAssign("obi_gnt_o", "obi_req_i & !obi_rsp_pending_q"),
                UGLIRAssign("obi_accept_n", "obi_req_i & obi_gnt_o"),
                UGLIRAssign("obi_rvalid_o", "obi_rsp_pending_q"),
                UGLIRAssign("obi_rdata_o", "obi_rsp_rdata_q"),
                UGLIRAssign("req_valid", "start_pending_q"),
                UGLIRAssign("resp_ready", "true"),
            ]
        )
    else:
        full_expr = _queue_full_expr(protocol)
        design.assigns.extend(
            [
                UGLIRAssign("obi_gnt_o", f"obi_req_i & !({full_expr})"),
                UGLIRAssign("obi_accept_n", "obi_req_i & obi_gnt_o"),
                UGLIRAssign("obi_rvalid_o", "obi_rsp_count_q != 0:u2"),
                UGLIRAssign("obi_rdata_o", "obi_rsp_fifo_q[obi_rsp_head_q]"),
                UGLIRAssign("req_valid", "start_pending_q"),
                UGLIRAssign("resp_ready", "true"),
            ]
        )
    design.assigns.append(UGLIRAssign("obi_hit_n", _obi_hit_expr(protocol)))
    for port in wrapper.scalar_inputs:
        design.assigns.append(UGLIRAssign(port.name, f"{port.name}_q"))
    for interface in wrapper.memory_interfaces:
        window = next(window for window in protocol.memory_windows if window.base == interface.base)
        design.assigns.append(
            UGLIRAssign(
                f"{interface.base}_bus_hit_n",
                f"obi_addr_i >= {window.symbol} && obi_addr_i < ({window.symbol} + {_u32(window.span_bytes)})",
            )
        )
        design.assigns.append(UGLIRAssign(f"{interface.base}_bus_word_addr_n", "obi_addr_i[11:2]"))
        design.assigns.append(
            UGLIRAssign(
                f"{interface.base}_rdata",
                f"{interface.base}_mem_q[{_memory_index_expr(f'{interface.base}_addr', interface)}]",
            )
        )

    seq = UGLIRSeqBlock(clock="clk", reset="rst")
    if protocol.response_queue_depth <= 1:
        seq.reset_updates.extend(
            [
                UGLIRSeqUpdate("obi_rsp_pending_q", "false"),
                UGLIRSeqUpdate("obi_rsp_rdata_q", "0:u32"),
                UGLIRSeqUpdate("start_pending_q", "false"),
                UGLIRSeqUpdate("busy_q", "false"),
                UGLIRSeqUpdate("done_q", "false"),
            ]
        )
    else:
        seq.reset_updates.extend(
            [
                UGLIRSeqUpdate("obi_rsp_head_q", "0:u1"),
                UGLIRSeqUpdate("obi_rsp_tail_q", "0:u1"),
                UGLIRSeqUpdate("obi_rsp_count_q", "0:u2"),
                UGLIRSeqUpdate("start_pending_q", "false"),
                UGLIRSeqUpdate("busy_q", "false"),
                UGLIRSeqUpdate("done_q", "false"),
            ]
        )
    for port in wrapper.scalar_inputs:
        seq.reset_updates.append(UGLIRSeqUpdate(f"{port.name}_q", _zero_expr_for_type(port.type)))
    for port in wrapper.scalar_outputs:
        seq.reset_updates.append(UGLIRSeqUpdate(f"{port.name}_q", _zero_expr_for_type(port.type)))

    control_write_cond = "(obi_accept_n && obi_we_i) && (obi_addr_i == OBI_REG_CONTROL_STATUS) && obi_wdata_i[0]"
    if protocol.response_queue_depth <= 1:
        seq.updates.append(
            UGLIRSeqUpdate(
                "obi_rsp_pending_q",
                "(obi_accept_n ? true : ((obi_rsp_pending_q && obi_rready_i) ? false : obi_rsp_pending_q))",
            )
        )
        seq.updates.append(
            UGLIRSeqUpdate(
                "obi_rsp_rdata_q",
                f"(obi_accept_n ? {_obi_read_data_expr(wrapper, protocol)} : obi_rsp_rdata_q)",
            )
        )
    else:
        pop_expr = "(obi_rsp_count_q != 0:u2) && obi_rready_i"
        seq.updates.append(
            UGLIRSeqUpdate(
                "obi_rsp_fifo_q[obi_rsp_tail_q]",
                _obi_read_data_expr(wrapper, protocol),
                "obi_accept_n",
            )
        )
        seq.updates.append(
            UGLIRSeqUpdate(
                "obi_rsp_head_q",
                f"({pop_expr} ? !obi_rsp_head_q : obi_rsp_head_q)",
            )
        )
        seq.updates.append(
            UGLIRSeqUpdate(
                "obi_rsp_tail_q",
                "(obi_accept_n ? !obi_rsp_tail_q : obi_rsp_tail_q)",
            )
        )
        seq.updates.append(
            UGLIRSeqUpdate(
                "obi_rsp_count_q",
                f"(obi_accept_n && !({pop_expr})) ? (obi_rsp_count_q + 1:u2) : (({pop_expr} && !obi_accept_n) ? (obi_rsp_count_q - 1:u2) : obi_rsp_count_q)",
            )
        )
    seq.updates.append(
        UGLIRSeqUpdate(
            "start_pending_q",
            f"({control_write_cond} && obi_be_i[0]) ? true : ((start_pending_q && req_ready) ? false : start_pending_q)",
        )
    )
    seq.updates.append(
        UGLIRSeqUpdate(
            "busy_q",
            "((start_pending_q && req_ready) ? true : (resp_valid ? false : busy_q))",
        )
    )
    seq.updates.append(
        UGLIRSeqUpdate(
            "done_q",
            f"({control_write_cond} && obi_be_i[0]) ? false : (resp_valid ? true : done_q)",
        )
    )
    for port in protocol.scalar_inputs:
        seq.updates.append(
            UGLIRSeqUpdate(
                f"{port.name}_q",
                _masked_write_expr(f"{port.name}_q", "obi_wdata_i", "obi_be_i", port.type),
                f"(obi_accept_n && obi_we_i) && (obi_addr_i == {port.symbol})",
            )
        )
    for port in wrapper.scalar_outputs:
        seq.updates.append(
            UGLIRSeqUpdate(
                f"{port.name}_q",
                f"resp_valid ? {port.name} : {port.name}_q",
            )
        )
    for interface in wrapper.memory_interfaces:
        seq.updates.append(
            UGLIRSeqUpdate(
                f"{interface.base}_mem_q[{interface.base}_bus_word_addr_n]",
                _masked_write_expr(
                    f"{interface.base}_mem_q[{interface.base}_bus_word_addr_n]",
                    "obi_wdata_i",
                    "obi_be_i",
                    interface.data_type,
                ),
                f"(obi_accept_n && obi_we_i) && {interface.base}_bus_hit_n",
            )
        )
        if interface.has_write:
            seq.updates.append(
                UGLIRSeqUpdate(
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
    design: UGLIRDesign,
    original_inputs: list[UGLIRPort],
    original_outputs: list[UGLIRPort],
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


def _dedupe_resources(design: UGLIRDesign) -> None:
    deduped: list[UGLIRResource] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for resource in design.resources:
        key = (resource.kind, resource.id, resource.value, resource.target)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resource)
    design.resources = deduped


def _dedupe_constants(design: UGLIRDesign) -> None:
    deduped: list[UGLIRConstant] = []
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


def _pack_to_obi(signal: str, type_hint: str) -> str:
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


def _memory_depth(interface) -> int:
    depth = getattr(interface, "depth", None)
    if isinstance(depth, int) and depth > 0:
        return depth
    return 1024


def _memory_index_width(interface) -> int:
    depth = _memory_depth(interface)
    return max(depth - 1, 1).bit_length()


def _queue_index_width(depth: int) -> int:
    return max(depth - 1, 1).bit_length()


def _queue_count_width(depth: int) -> int:
    return max(depth, 1).bit_length()


def _queue_full_expr(protocol: OBISlaveProtocolPlan) -> str:
    return f"obi_rsp_count_q == {_queue_count_literal(protocol.response_queue_depth, _queue_count_width(protocol.response_queue_depth))}"


def _queue_count_literal(value: int, width: int) -> str:
    return f"{value}:u{width}"


def _memory_index_expr(address_expr: str, interface) -> str:
    width = _memory_index_width(interface)
    if width <= 1:
        return f"{address_expr}[0]"
    return f"{address_expr}[{width - 1}:0]"


def _obi_read_data_expr(wrapper: SlaveWrapperPlan, protocol: OBISlaveProtocolPlan) -> str:
    read_expr = "0:u32"
    for interface in reversed(wrapper.memory_interfaces):
        read_expr = (
            f"{interface.base}_bus_hit_n ? "
            f"{_pack_to_obi(f'{interface.base}_mem_q[{interface.base}_bus_word_addr_n]', interface.data_type)} : "
            f"({read_expr})"
        )
    for port in reversed(protocol.scalar_outputs):
        read_expr = f"(obi_addr_i == {port.symbol}) ? {_pack_to_obi(port.name + '_q', port.type)} : ({read_expr})"
    for port in reversed(protocol.scalar_inputs):
        read_expr = f"(obi_addr_i == {port.symbol}) ? {_pack_to_obi(port.name + '_q', port.type)} : ({read_expr})"
    status_expr = "{0:u28, req_ready, start_pending_q, busy_q, done_q}"
    read_expr = f"(obi_addr_i == OBI_REG_CONTROL_STATUS) ? {status_expr} : ({read_expr})"
    return f"(obi_we_i ? 0:u32 : ({read_expr}))"


def _obi_hit_expr(protocol: OBISlaveProtocolPlan) -> str:
    terms = ["(obi_addr_i == OBI_REG_CONTROL_STATUS)"]
    terms.extend(f"(obi_addr_i == {port.symbol})" for port in protocol.scalar_inputs)
    terms.extend(f"(obi_addr_i == {port.symbol})" for port in protocol.scalar_outputs)
    terms.extend(f"{window.base}_bus_hit_n" for window in protocol.memory_windows)
    return " || ".join(terms) if terms else "false"


def _obi_address_map(protocol: OBISlaveProtocolPlan) -> UGLIRAddressMap:
    address_map = UGLIRAddressMap("obi")
    address_map.entries.append(
        UGLIRAddressMapEntry(
            "register",
            "control_status",
            {
                "offset": _u32(protocol.control_status_address),
                "access": "rw",
                "symbol": "OBI_REG_CONTROL_STATUS",
            },
        )
    )
    for port in protocol.scalar_inputs:
        address_map.entries.append(
            UGLIRAddressMapEntry(
                "register",
                port.name,
                {
                    "offset": _u32(port.address),
                    "access": "rw",
                    "symbol": port.symbol,
                    "type": port.type,
                },
            )
        )
    for port in protocol.scalar_outputs:
        address_map.entries.append(
            UGLIRAddressMapEntry(
                "register",
                port.name,
                {
                    "offset": _u32(port.address),
                    "access": "ro",
                    "symbol": port.symbol,
                    "type": port.type,
                },
            )
        )
    for window in protocol.memory_windows:
        address_map.entries.append(
            UGLIRAddressMapEntry(
                "memory",
                window.base,
                {
                    "offset": _u32(window.base_address),
                    "span": _u32(window.span_bytes),
                    "access": "rw",
                    "symbol": window.symbol,
                    "word_t": window.data_type,
                    "depth": _memory_depth(window),
                },
            )
        )
    return address_map


def _masked_write_expr(current_expr: str, write_data_expr: str, be_expr: str, type_hint: str) -> str:
    match = re.fullmatch(r"([iu])(\d+)", type_hint)
    if match is None:
        return write_data_expr
    width = int(match.group(2))
    if width <= 0:
        return current_expr
    if width == 1:
        return f"({be_expr}[0] ? {write_data_expr}[0] : {current_expr})"

    parts: list[str] = []
    byte_count = max((width + 7) // 8, 1)
    for byte_index in reversed(range(byte_count)):
        lo = byte_index * 8
        hi = min(width - 1, lo + 7)
        current_slice = _bit_slice_expr(current_expr, hi, lo)
        write_slice = _bit_slice_expr(write_data_expr, hi, lo)
        parts.append(f"({be_expr}[{byte_index}] ? {write_slice} : {current_slice})")
    if len(parts) == 1:
        return parts[0]
    return "{" + ", ".join(parts) + "}"


def _bit_slice_expr(signal: str, hi: int, lo: int) -> str:
    if hi == lo:
        return f"{signal}[{hi}]"
    return f"{signal}[{hi}:{lo}]"
