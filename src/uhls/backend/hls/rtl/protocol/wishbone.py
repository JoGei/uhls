"""Generic Wishbone protocol planning shared across RTL backends."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any

from uhls.backend.hls.uglir.model import (
    UGLIRAddressMap,
    UGLIRAddressMapEntry,
    UGLIRAssign,
    UGLIRAttach,
    UGLIRConstant,
    UGLIRDesign,
    UGLIRPort,
    UGLIRResource,
    UGLIRSeqBlock,
    UGLIRSeqUpdate,
)

from ..wrap import SlaveWrapperPlan
from .memory_component import (
    component_memory_control_literals,
    component_memory_port_name,
    component_memory_port_type,
    component_semantic_port_signal,
    component_tied_input_ports,
    materialized_memory_instance_spec,
    resolved_component_ports,
)


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
    features: tuple[str, ...] = ()
    err_terminate: bool = False


def plan_wishbone_slave_protocol(wrapper: SlaveWrapperPlan, features: tuple[str, ...] = ()) -> WishboneSlaveProtocolPlan:
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
    err_terminate = "err" in features
    ports = [
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
    ]
    if err_terminate:
        ports.append(ProtocolPort("output", "i1", "wb_err_o"))
    return WishboneSlaveProtocolPlan(
        ports=tuple(ports),
        control_status_address=0x0000,
        scalar_inputs=scalar_inputs,
        scalar_outputs=scalar_outputs,
        memory_windows=memory_windows,
        features=features,
        err_terminate=err_terminate,
    )


def build_wishbone_slave_wrapper_uglir(
    core_design: UGLIRDesign,
    wrapper: SlaveWrapperPlan,
    protocol: WishboneSlaveProtocolPlan,
    *,
    component_library: dict[str, dict[str, Any]] | None = None,
) -> UGLIRDesign:
    """Merge a Wishbone slave wrapper around one core µglIR design."""
    design = deepcopy(core_design)
    if design.stage != "uglir":
        raise ValueError(f"wishbone wrapper builder expects µglIR input, got stage '{design.stage}'")

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

    design.constants.append(UGLIRConstant("WB_REG_CONTROL_STATUS", f"WB_BASE_ADDR + {_u32(protocol.control_status_address)}", "u32"))
    for port in protocol.scalar_inputs:
        design.constants.append(UGLIRConstant(port.symbol, f"WB_BASE_ADDR + {_u32(port.address)}", "u32"))
    for port in protocol.scalar_outputs:
        design.constants.append(UGLIRConstant(port.symbol, f"WB_BASE_ADDR + {_u32(port.address)}", "u32"))
    for window in protocol.memory_windows:
        design.constants.append(UGLIRConstant(window.symbol, f"WB_BASE_ADDR + {_u32(window.base_address)}", "u32"))
    design.address_maps.append(_wishbone_address_map(protocol))

    slow_memory_bases = tuple(
        interface.base
        for interface in wrapper.memory_interfaces
        if _is_slow_instantiated_memory(interface, component_library)
    )

    design.resources.extend(
        [
            UGLIRResource("net", "wb_req_n", "i1"),
            UGLIRResource("net", "wb_hit_n", "i1"),
            UGLIRResource("reg", "start_pending_q", "i1"),
            UGLIRResource("reg", "busy_q", "i1"),
            UGLIRResource("reg", "done_q", "i1"),
        ]
    )
    for port in wrapper.scalar_inputs:
        design.resources.append(UGLIRResource("reg", f"{port.name}_q", port.type))
    for port in wrapper.scalar_outputs:
        design.resources.append(UGLIRResource("reg", f"{port.name}_q", port.type))
    for interface in wrapper.memory_interfaces:
        design.resources.append(UGLIRResource("net", f"{interface.base}_bus_hit_n", "i1"))
        design.resources.append(UGLIRResource("net", f"{interface.base}_bus_word_addr_n", f"u{_memory_index_width(interface)}"))
        if _should_instantiate_memory_component(interface, component_library):
            design.resources.append(UGLIRResource("net", f"{interface.base}_mem_rdata_n", interface.data_type))
            design.resources.append(
                UGLIRResource(
                    "inst",
                    f"{interface.base}_mem_inst",
                    _materialized_interface_component_spec(interface, component_library),
                )
            )
        else:
            depth = _memory_depth(interface)
            design.resources.append(UGLIRResource("mem", f"{interface.base}_mem_q", f"{interface.data_type}[{depth}]"))
        if interface.base in slow_memory_bases:
            design.resources.append(UGLIRResource("reg", f"{interface.base}_bus_read_pending_q", "i1"))

    design.assigns.extend(
        [
            UGLIRAssign("wb_req_n", "wb_cyc_i & wb_stb_i"),
            UGLIRAssign("req_valid", "start_pending_q"),
            UGLIRAssign("resp_ready", "true"),
            UGLIRAssign("wb_dat_o", _wishbone_read_data_expr(wrapper, protocol, component_library)),
        ]
    )
    design.assigns.append(UGLIRAssign("wb_hit_n", _wishbone_hit_expr(protocol)))
    if protocol.err_terminate:
        design.assigns.append(
            UGLIRAssign("wb_ack_o", _wishbone_ack_expr(wrapper, protocol, component_library))
        )
        design.assigns.append(UGLIRAssign("wb_err_o", "wb_req_n & !wb_hit_n"))
    else:
        design.assigns.append(
            UGLIRAssign("wb_ack_o", _wishbone_ack_expr(wrapper, protocol, component_library, require_hit=False))
        )
    for port in wrapper.scalar_inputs:
        design.assigns.append(UGLIRAssign(port.name, f"{port.name}_q"))
    for interface in wrapper.memory_interfaces:
        window = next(window for window in protocol.memory_windows if window.base == interface.base)
        design.assigns.append(
            UGLIRAssign(
                f"{interface.base}_bus_hit_n",
                (
                    f"wb_adr_i >= {window.symbol} && wb_adr_i < ({window.symbol} + {_u32(window.span_bytes)})"
                    if not _should_instantiate_memory_component(interface, component_library)
                    else f"(wb_adr_i >= {window.symbol} && wb_adr_i < ({window.symbol} + {_u32(window.span_bytes)})) && !(busy_q | start_pending_q)"
                ),
            )
        )
        design.assigns.append(
            UGLIRAssign(
                f"{interface.base}_bus_word_addr_n",
                f"((wb_adr_i - {window.symbol}) >> 2)",
            )
        )
        design.assigns.append(
            UGLIRAssign(
                f"{interface.base}_rdata",
                (
                    f"{interface.base}_mem_rdata_n"
                    if _should_instantiate_memory_component(interface, component_library)
                    else f"{interface.base}_mem_q[{_memory_index_expr(f'{interface.base}_addr', interface)}]"
                ),
            )
        )
        if _should_instantiate_memory_component(interface, component_library):
            core_write_expr = f"{interface.base}_we" if interface.has_write else "false"
            core_wdata_expr = (
                f"{interface.base}_wdata"
                if interface.has_write and interface.write_type is not None
                else _zero_expr_for_type(interface.data_type)
            )
            _append_instantiated_wrapper_memory(
                design,
                interface,
                component_library,
                addr_expr=f"((busy_q | start_pending_q) ? {interface.base}_addr : {interface.base}_bus_word_addr_n)",
                wdata_expr=f"((busy_q | start_pending_q) ? {core_wdata_expr} : {_unpack_from_wishbone('wb_dat_i', interface.data_type)})",
                active_expr=f"((busy_q | start_pending_q) || (wb_req_n && {interface.base}_bus_hit_n))",
                write_expr=f"((busy_q | start_pending_q) ? {core_write_expr} : ((wb_req_n && wb_we_i) && {interface.base}_bus_hit_n))",
            )

    seq = UGLIRSeqBlock(clock="clk", reset="rst")
    seq.reset_updates.extend(
        [
            UGLIRSeqUpdate("start_pending_q", "false"),
            UGLIRSeqUpdate("busy_q", "false"),
            UGLIRSeqUpdate("done_q", "false"),
        ]
    )
    for base in slow_memory_bases:
        seq.reset_updates.append(UGLIRSeqUpdate(f"{base}_bus_read_pending_q", "false"))
    for port in wrapper.scalar_inputs:
        seq.reset_updates.append(UGLIRSeqUpdate(f"{port.name}_q", _zero_expr_for_type(port.type)))
    for port in wrapper.scalar_outputs:
        seq.reset_updates.append(UGLIRSeqUpdate(f"{port.name}_q", _zero_expr_for_type(port.type)))

    control_write_cond = "(wb_req_n && wb_we_i) && (wb_adr_i == WB_REG_CONTROL_STATUS) && wb_dat_i[0]"
    seq.updates.append(
        UGLIRSeqUpdate(
            "start_pending_q",
            f"({control_write_cond} && wb_sel_i[0]) ? true : ((start_pending_q && req_ready) ? false : start_pending_q)",
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
            f"({control_write_cond} && wb_sel_i[0]) ? false : (resp_valid ? true : done_q)",
        )
    )
    for port in protocol.scalar_inputs:
        seq.updates.append(
            UGLIRSeqUpdate(
                f"{port.name}_q",
                _wishbone_masked_write_expr(f"{port.name}_q", "wb_dat_i", "wb_sel_i", port.type),
                f"(wb_req_n && wb_we_i) && (wb_adr_i == {port.symbol})",
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
        if interface.base in slow_memory_bases:
            seq.updates.append(
                UGLIRSeqUpdate(
                    f"{interface.base}_bus_read_pending_q",
                    f"{interface.base}_bus_read_pending_q ? false : ((wb_req_n && !wb_we_i) && {interface.base}_bus_hit_n)",
                )
            )
        if not _should_instantiate_memory_component(interface, component_library):
            seq.updates.append(
                UGLIRSeqUpdate(
                    f"{interface.base}_mem_q[{interface.base}_bus_word_addr_n]",
                    _wishbone_masked_write_expr(
                        f"{interface.base}_mem_q[{interface.base}_bus_word_addr_n]",
                        "wb_dat_i",
                        "wb_sel_i",
                        interface.data_type,
                    ),
                    f"(wb_req_n && wb_we_i) && {interface.base}_bus_hit_n",
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


def _wishbone_read_data_expr(
    wrapper: SlaveWrapperPlan,
    protocol: WishboneSlaveProtocolPlan,
    component_library: dict[str, dict[str, Any]] | None = None,
) -> str:
    read_expr = "0:u32"
    for interface in reversed(wrapper.memory_interfaces):
        if _is_slow_instantiated_memory(interface, component_library):
            read_expr = (
                f"{interface.base}_bus_read_pending_q ? "
                f"{_pack_to_wishbone(f'{interface.base}_mem_rdata_n', interface.data_type)} : "
                f"({read_expr})"
            )
            continue
        source_expr = (
            f"{interface.base}_mem_rdata_n"
            if _should_instantiate_memory_component(interface, component_library)
            else f"{interface.base}_mem_q[{interface.base}_bus_word_addr_n]"
        )
        read_expr = (
            f"{interface.base}_bus_hit_n ? "
            f"{_pack_to_wishbone(source_expr, interface.data_type)} : "
            f"({read_expr})"
        )
    for port in reversed(protocol.scalar_outputs):
        read_expr = f"(wb_adr_i == {port.symbol}) ? {_pack_to_wishbone(port.name + '_q', port.type)} : ({read_expr})"
    for port in reversed(protocol.scalar_inputs):
        read_expr = f"(wb_adr_i == {port.symbol}) ? {_pack_to_wishbone(port.name + '_q', port.type)} : ({read_expr})"
    status_expr = "{0:u28, req_ready, start_pending_q, busy_q, done_q}"
    read_expr = f"(wb_adr_i == WB_REG_CONTROL_STATUS) ? {status_expr} : ({read_expr})"
    return f"((wb_req_n && !wb_we_i) ? ({read_expr}) : 0:u32)"


def _wishbone_ack_expr(
    wrapper: SlaveWrapperPlan,
    protocol: WishboneSlaveProtocolPlan,
    component_library: dict[str, dict[str, Any]] | None = None,
    *,
    require_hit: bool = True,
) -> str:
    request_expr = "wb_req_n" if not require_hit else "wb_req_n & wb_hit_n"
    slow_read_expr = _wishbone_slow_memory_read_hit_expr(wrapper, component_library)
    pending_expr = _wishbone_slow_memory_pending_expr(wrapper, component_library)
    if slow_read_expr == "false" and pending_expr == "false":
        return request_expr
    return f"(({request_expr}) & !({slow_read_expr})) | ({pending_expr})"


def _wishbone_hit_expr(protocol: WishboneSlaveProtocolPlan) -> str:
    terms = ["(wb_adr_i == WB_REG_CONTROL_STATUS)"]
    terms.extend(f"(wb_adr_i == {port.symbol})" for port in protocol.scalar_inputs)
    terms.extend(f"(wb_adr_i == {port.symbol})" for port in protocol.scalar_outputs)
    terms.extend(f"{window.base}_bus_hit_n" for window in protocol.memory_windows)
    return " || ".join(terms) if terms else "false"


def _wishbone_slow_memory_read_hit_expr(
    wrapper: SlaveWrapperPlan,
    component_library: dict[str, dict[str, Any]] | None,
) -> str:
    terms = [
        f"{interface.base}_bus_hit_n"
        for interface in wrapper.memory_interfaces
        if _is_slow_instantiated_memory(interface, component_library)
    ]
    if not terms:
        return "false"
    return f"(!wb_we_i) & ({' | '.join(terms)})"


def _wishbone_slow_memory_pending_expr(
    wrapper: SlaveWrapperPlan,
    component_library: dict[str, dict[str, Any]] | None,
) -> str:
    terms = [
        f"{interface.base}_bus_read_pending_q"
        for interface in wrapper.memory_interfaces
        if _is_slow_instantiated_memory(interface, component_library)
    ]
    return " | ".join(terms) if terms else "false"


def _wishbone_address_map(protocol: WishboneSlaveProtocolPlan) -> UGLIRAddressMap:
    address_map = UGLIRAddressMap("wishbone")
    address_map.entries.append(
        UGLIRAddressMapEntry(
            "register",
            "control_status",
            {
                "offset": _u32(protocol.control_status_address),
                "access": "rw",
                "symbol": "WB_REG_CONTROL_STATUS",
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
        access = "rw"
        if protocol.err_terminate:
            access = f"{access}_err"
        address_map.entries.append(
            UGLIRAddressMapEntry(
                "memory",
                window.base,
                {
                    "offset": _u32(window.base_address),
                    "span": _u32(window.span_bytes),
                    "access": access,
                    "symbol": window.symbol,
                    "word_t": window.data_type,
                    "depth": _memory_depth(window),
                },
            )
        )
    return address_map


def _wishbone_masked_write_expr(current_expr: str, write_data_expr: str, select_expr: str, type_hint: str) -> str:
    match = re.fullmatch(r"([iu])(\d+)", type_hint)
    if match is None:
        return write_data_expr
    width = int(match.group(2))
    if width <= 0:
        return current_expr
    if width == 1:
        return f"({select_expr}[0] ? {write_data_expr}[0] : {current_expr})"

    parts: list[str] = []
    byte_count = max((width + 7) // 8, 1)
    for byte_index in reversed(range(byte_count)):
        lo = byte_index * 8
        hi = min(width - 1, lo + 7)
        current_slice = _bit_slice_expr(current_expr, hi, lo)
        write_slice = _bit_slice_expr(write_data_expr, hi, lo)
        parts.append(f"({select_expr}[{byte_index}] ? {write_slice} : {current_slice})")
    if len(parts) == 1:
        return parts[0]
    return "{" + ", ".join(parts) + "}"


def _bit_slice_expr(signal: str, hi: int, lo: int) -> str:
    if hi == lo:
        return f"{signal}[{lo}]"
    return f"{signal}[{hi}:{lo}]"


def _should_instantiate_memory_component(
    interface,
    component_library: dict[str, dict[str, Any]] | None,
) -> bool:
    return component_library is not None and interface.component_spec is not None


def _is_slow_instantiated_memory(
    interface,
    component_library: dict[str, dict[str, Any]] | None,
) -> bool:
    return _should_instantiate_memory_component(interface, component_library) and getattr(interface, "load_delay", 1) > 1


def _materialized_interface_component_spec(
    interface,
    component_library: dict[str, dict[str, Any]] | None,
) -> str:
    if component_library is None or interface.component_spec is None:
        raise ValueError(f"wrapper memory '{interface.base}' requires one component library for instantiation")
    return materialized_memory_instance_spec(component_library, interface.component_spec)


def _append_instantiated_wrapper_memory(
    design: UGLIRDesign,
    interface,
    component_library: dict[str, dict[str, Any]] | None,
    *,
    addr_expr: str,
    wdata_expr: str,
    active_expr: str,
    write_expr: str,
) -> None:
    if component_library is None or interface.component_spec is None:
        raise ValueError(f"wrapper memory '{interface.base}' requires one component library for instantiation")
    instance_id = f"{interface.base}_mem_inst"
    ports = resolved_component_ports(component_library, interface.component_spec)
    ties = component_tied_input_ports(component_library, interface.component_spec)
    addr_port = component_memory_port_name(component_library, interface.component_spec, "addr")
    wdata_port = component_memory_port_name(component_library, interface.component_spec, "wdata")
    rdata_port = component_memory_port_name(component_library, interface.component_spec, "rdata")
    control_literals = component_memory_control_literals(component_library, interface.component_spec)
    if rdata_port is None:
        raise ValueError(f"wrapper memory '{interface.base}' must define one readable output port")

    for port_name, payload in ports.items():
        port_type = payload.get("type")
        if not isinstance(port_type, str) or not port_type:
            continue
        semantic_signal = component_semantic_port_signal(component_library, interface.component_spec, port_name)
        if semantic_signal is not None:
            design.attachments.append(UGLIRAttach(instance_id, port_name, semantic_signal))
            continue
        if payload.get("dir") == "output":
            if port_name == rdata_port:
                design.attachments.append(UGLIRAttach(instance_id, port_name, f"{interface.base}_mem_rdata_n"))
            continue
        signal_name = f"{instance_id}_{port_name}_n"
        design.resources.append(UGLIRResource("net", signal_name, port_type))
        design.attachments.append(UGLIRAttach(instance_id, port_name, signal_name))
        if port_name in ties:
            design.assigns.append(UGLIRAssign(signal_name, ties[port_name]))
            continue
        if port_name == addr_port:
            design.assigns.append(UGLIRAssign(signal_name, addr_expr))
            continue
        if port_name == wdata_port:
            design.assigns.append(UGLIRAssign(signal_name, wdata_expr))
            continue
        load_expr, store_expr = control_literals.get(port_name, (None, None))
        load_expr = load_expr or _zero_expr_for_type(port_type)
        store_expr = store_expr or _zero_expr_for_type(port_type)
        design.assigns.append(
            UGLIRAssign(
                signal_name,
                f"({active_expr}) ? ({write_expr} ? ({store_expr}) : ({load_expr})) : ({_zero_expr_for_type(port_type)})",
            )
        )
