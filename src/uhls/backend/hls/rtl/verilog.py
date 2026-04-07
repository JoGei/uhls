"""Verilog emission for uglir designs."""

from __future__ import annotations

from math import ceil, log2
import re

from uhls.backend.hls.uhir.model import UHIRDesign, UHIRGlueMux, UHIRPort, UHIRResource, UHIRSeqBlock, UHIRSeqUpdate
from .protocol import WishboneSlaveProtocolPlan, plan_wishbone_slave_protocol
from .wrap import SlaveWrapperPlan, plan_master_wrapper, plan_slave_wrapper, wrapper_core_signal

_TYPED_INT_RE = re.compile(r"(?<![\w$])(-?\d+):(i|u)(\d+)\b")
_IDENT_RE = re.compile(r"\b[A-Za-z_][\w$]*\b")
_VERILOG_KEYWORDS = frozenset({
    "always",
    "and",
    "assign",
    "begin",
    "case",
    "default",
    "else",
    "end",
    "endmodule",
    "if",
    "input",
    "localparam",
    "module",
    "or",
    "output",
    "reg",
    "wire",
})


def emit_uglir_to_verilog(design: UHIRDesign, module_name: str | None = None) -> str:
    """Render one uglir design as Verilog."""
    if design.stage != "uglir":
        raise ValueError(f"verilog emission expects uglir input, got stage '{design.stage}'")
    rendered_module_name = design.name if module_name is None else module_name

    ctrl_widths = _ctrl_widths(design)
    ctrl_enums = _ctrl_enum_symbols(design)
    signal_types = _signal_types(design)

    lines: list[str] = []
    port_lines = [f"    {_format_port_decl(port, ctrl_widths)}" for port in [*design.inputs, *design.outputs]]
    lines.append(f"module {rendered_module_name} (")
    for index, port_line in enumerate(port_lines):
        suffix = "," if index < len(port_lines) - 1 else ""
        lines.append(f"{port_line}{suffix}")
    lines.append(");")
    lines.append("")

    if design.constants:
        for const_decl in design.constants:
            lines.append(
                f"  {_join_decl_tokens('localparam', _format_decl_type(const_decl.type, ctrl_widths), const_decl.name)} = "
                f"{_translate_expr(str(const_decl.value), ctrl_enums, None)};"
            )
        lines.append("")

    for select_signal, symbol_map in ctrl_enums.items():
        width = ctrl_widths.get(select_signal, 1)
        for index, key in enumerate(_ctrl_keys_for_signal(design, select_signal)):
            symbol = symbol_map[key]
            lines.append(
                f"  {_join_decl_tokens('localparam', _format_width(width), symbol)} = "
                f"{_format_literal(index, width, signed=False)};"
            )
        lines.append("")

    top_level_signals = {port.name for port in [*design.inputs, *design.outputs]}
    for resource in design.resources:
        if resource.kind == "inst" or resource.kind == "port":
            continue
        if resource.id in top_level_signals:
            continue
        if resource.kind == "reg":
            lines.append(f"  {_join_decl_tokens('reg', _format_decl_type(resource.value, ctrl_widths, resource.id), resource.id)};")
            continue
        if resource.kind in {"net", "mux"}:
            lines.append(f"  {_join_decl_tokens('wire', _format_decl_type(resource.value, ctrl_widths, resource.id), resource.id)};")
            continue
        raise ValueError(f"unsupported uglir resource '{resource.kind}' during verilog emission")
    if any(resource.kind not in {"inst", "port"} and resource.id not in top_level_signals for resource in design.resources):
        lines.append("")

    for assign in design.assigns:
        target_type = signal_types.get(assign.target)
        lines.append(
            f"  assign {assign.target} = {_translate_expr(assign.expr, ctrl_enums, assign.target if target_type == 'ctrl' else None)};"
        )
    if design.assigns:
        lines.append("")

    for glue_mux in design.glue_muxes:
        lines.append(f"  assign {glue_mux.name} = {_mux_expr(glue_mux, ctrl_enums)};")
    if design.glue_muxes:
        lines.append("")

    attachments_by_instance: dict[str, list[tuple[str, str]]] = {}
    for attachment in design.attachments:
        attachments_by_instance.setdefault(attachment.instance, []).append((attachment.port, attachment.signal))
    for resource in design.resources:
        if resource.kind != "inst":
            continue
        lines.append(f"  {resource.value} {resource.id} (")
        instance_ports = attachments_by_instance.get(resource.id, [])
        for index, (port_name, signal_name) in enumerate(instance_ports):
            suffix = "," if index < len(instance_ports) - 1 else ""
            lines.append(f"    .{port_name}({signal_name}){suffix}")
        lines.append("  );")
        lines.append("")

    for seq_block in design.seq_blocks:
        lines.extend(_format_seq_block(seq_block, ctrl_enums))
        lines.append("")

    if lines[-1] == "":
        lines.pop()
    lines.append("endmodule")
    return "\n".join(lines)


def emit_uglir_to_verilog_wrapped(design: UHIRDesign, wrap: str, protocol: str) -> str:
    """Render one uglir design plus one protocol wrapper."""
    if wrap == "slave" and protocol == "wishbone":
        wrapper_plan = plan_slave_wrapper(design, protocol)
        protocol_plan = plan_wishbone_slave_protocol(wrapper_plan)
    elif wrap == "master":
        plan_master_wrapper(design, protocol)
        raise NotImplementedError(f"verilog wrapper stub for wrap='{wrap}' protocol='{protocol}' is not implemented yet")
    else:
        raise NotImplementedError(f"verilog wrapper stub for wrap='{wrap}' protocol='{protocol}' is not implemented yet")

    core_module_name = f"{design.name}_core"
    wrapper_lines = [
        f"// Wrapper for wrap={wrap} protocol={protocol}.",
        "// Register map:",
        "//   0x0000 control/status: write bit0=start, read bit0=done bit1=busy bit2=start_pending bit3=req_ready",
        "//   0x0100.. scalar input registers (4-byte stride)",
        "//   0x0200.. scalar output registers (4-byte stride)",
        "//   0x1000.. memory windows (4 KiB per memory, word addressed)",
        emit_uglir_to_verilog(design, module_name=core_module_name),
        "",
        f"module {design.name} #(",
        "    parameter [31:0] WB_BASE_ADDR = 32'h0000_0000",
        ") (",
        *[
            f"    {_format_port_decl(UHIRPort(port.direction, port.name, port.type), {})}{',' if index < len(protocol_plan.ports) - 1 else ''}"
            for index, port in enumerate(protocol_plan.ports)
        ],
        ");",
        "",
    ]
    wrapper_lines.extend(_emit_wishbone_slave_wrapper_body(design, core_module_name, wrapper_plan, protocol_plan))
    wrapper_lines.append("endmodule")
    return "\n".join(wrapper_lines)


def _format_port_decl(port: UHIRPort, ctrl_widths: dict[str, int]) -> str:
    return _join_decl_tokens(port.direction, _format_decl_type(port.type, ctrl_widths, port.name), port.name)


def _signal_types(design: UHIRDesign) -> dict[str, str]:
    signal_types: dict[str, str] = {}
    for port in [*design.inputs, *design.outputs]:
        signal_types[port.name] = port.type
    for resource in design.resources:
        if resource.kind in {"reg", "net", "mux"}:
            signal_types[resource.id] = resource.value
    return signal_types


def _ctrl_widths(design: UHIRDesign) -> dict[str, int]:
    widths: dict[str, int] = {}
    for glue_mux in design.glue_muxes:
        widths[glue_mux.select] = max(1, ceil(log2(max(len(glue_mux.cases), 1))))
    return widths


def _ctrl_keys_for_signal(design: UHIRDesign, select_signal: str) -> list[str]:
    for glue_mux in design.glue_muxes:
        if glue_mux.select == select_signal:
            return [case.key for case in glue_mux.cases]
    return []


def _ctrl_enum_symbols(design: UHIRDesign) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    for glue_mux in design.glue_muxes:
        signal_map: dict[str, str] = {}
        for case in glue_mux.cases:
            signal_map[case.key] = f"{_sanitize_identifier(glue_mux.select)}_{_sanitize_identifier(case.key)}"
        mappings[glue_mux.select] = signal_map
    return mappings


def _format_decl_type(type_hint: str, ctrl_widths: dict[str, int], signal_name: str | None = None) -> str:
    if type_hint == "clock":
        return ""
    if type_hint == "ctrl":
        width = ctrl_widths.get(signal_name or "", 1)
        return _format_width(width)
    match = re.fullmatch(r"([iu])(\d+)", type_hint)
    if match is None:
        return ""
    signed = match.group(1) == "i"
    width = int(match.group(2))
    if width <= 1:
        return ""
    return f"{'signed ' if signed else ''}{_format_width(width)}"


def _format_width(width: int) -> str:
    return "" if width <= 1 else f"[{width - 1}:0]"


def _format_literal(value: int, width: int, signed: bool) -> str:
    prefix = "s" if signed else ""
    if value < 0:
        return f"-{width}'{prefix}d{abs(value)}"
    return f"{width}'{prefix}d{value}"


def _format_hex32(value: int) -> str:
    upper = (value >> 16) & 0xFFFF
    lower = value & 0xFFFF
    return f"32'h{upper:04x}_{lower:04x}"


def _translate_expr(expr: str, ctrl_enums: dict[str, dict[str, str]], ctrl_target: str | None) -> str:
    translated = expr
    translated = re.sub(r"\btrue\b", "1'b1", translated)
    translated = re.sub(r"\bfalse\b", "1'b0", translated)

    def replace_typed_int(match: re.Match[str]) -> str:
        value = int(match.group(1))
        signed = match.group(2) == "i"
        width = int(match.group(3))
        return _format_literal(value, width, signed)

    translated = _TYPED_INT_RE.sub(replace_typed_int, translated)

    if ctrl_target is not None and ctrl_target in ctrl_enums:
        enum_map = ctrl_enums[ctrl_target]

        def replace_enum_token(match: re.Match[str]) -> str:
            token = match.group(0)
            return enum_map.get(token, token)

        translated = _IDENT_RE.sub(replace_enum_token, translated)

    return translated


def _mux_expr(glue_mux: UHIRGlueMux, ctrl_enums: dict[str, dict[str, str]]) -> str:
    symbol_map = ctrl_enums.get(glue_mux.select, {})
    translated_sources = [_translate_expr(case.source, ctrl_enums, None) for case in glue_mux.cases]
    if not glue_mux.cases:
        raise ValueError(f"uglir mux '{glue_mux.name}' must declare at least one case")
    expr = translated_sources[-1]
    for case, source in reversed(list(zip(glue_mux.cases[:-1], translated_sources[:-1], strict=False))):
        expr = f"({glue_mux.select} == {symbol_map[case.key]}) ? {source} : {expr}"
    return expr


def _format_seq_block(seq_block: UHIRSeqBlock, ctrl_enums: dict[str, dict[str, str]]) -> list[str]:
    lines = [f"  always @(posedge {seq_block.clock}) begin"]
    if seq_block.reset is not None:
        lines.append(f"    if ({_translate_expr(seq_block.reset, ctrl_enums, None)}) begin")
        for update in seq_block.reset_updates:
            lines.append(f"      {_format_seq_update(update, ctrl_enums)}")
        lines.append("    end else begin")
        lines.extend(_format_seq_updates(seq_block.updates, ctrl_enums, indent="      "))
        lines.append("    end")
    else:
        lines.extend(_format_seq_updates(seq_block.updates, ctrl_enums, indent="    "))
    lines.append("  end")
    return lines


def _format_seq_updates(updates: list[UHIRSeqUpdate], ctrl_enums: dict[str, dict[str, str]], indent: str) -> list[str]:
    lines: list[str] = []
    for update in updates:
        if update.enable is None:
            lines.append(f"{indent}{_format_seq_update(update, ctrl_enums)}")
            continue
        lines.append(f"{indent}if ({_translate_expr(update.enable, ctrl_enums, None)}) begin")
        lines.append(f"{indent}  {_format_seq_update(UHIRSeqUpdate(update.target, update.value), ctrl_enums)}")
        lines.append(f"{indent}end")
    return lines


def _format_seq_update(update: UHIRSeqUpdate, ctrl_enums: dict[str, dict[str, str]]) -> str:
    return f"{update.target} <= {_translate_expr(update.value, ctrl_enums, None)};"


def _emit_wishbone_slave_wrapper_body(
    design: UHIRDesign,
    core_module_name: str,
    wrapper_plan: SlaveWrapperPlan,
    protocol_plan: WishboneSlaveProtocolPlan,
) -> list[str]:
    scalar_inputs = list(wrapper_plan.scalar_inputs)
    scalar_outputs = list(wrapper_plan.scalar_outputs)
    memory_interfaces = list(wrapper_plan.memory_interfaces)
    lines: list[str] = []
    lines.append(
        f"  localparam [31:0] WB_REG_CONTROL_STATUS = WB_BASE_ADDR + {_format_hex32(protocol_plan.control_status_address)};"
    )
    for port in protocol_plan.scalar_inputs:
        lines.append(f"  localparam [31:0] {port.symbol} = WB_BASE_ADDR + {_format_hex32(port.address)};")
    for port in protocol_plan.scalar_outputs:
        lines.append(f"  localparam [31:0] {port.symbol} = WB_BASE_ADDR + {_format_hex32(port.address)};")
    for window in protocol_plan.memory_windows:
        lines.append(f"  localparam [31:0] {window.symbol} = WB_BASE_ADDR + {_format_hex32(window.base_address)};")
    if scalar_inputs or scalar_outputs or memory_interfaces:
        lines.append("")

    lines.extend(
        [
            "  wire wb_req_n;",
            "  wire core_req_valid_n;",
            "  wire core_resp_ready_n;",
            "  wire core_req_ready_n;",
            "  wire core_resp_valid_n;",
            "  reg start_pending_q;",
            "  reg busy_q;",
            "  reg done_q;",
            "  reg [31:0] wb_dat_o_q;",
            "",
            "  assign wb_req_n = wb_cyc_i & wb_stb_i;",
            "  assign wb_ack_o = wb_req_n;",
            "  assign wb_dat_o = wb_dat_o_q;",
            "  assign core_req_valid_n = start_pending_q;",
            "  assign core_resp_ready_n = 1'b1;",
            "",
        ]
    )

    for port in scalar_inputs:
        lines.append(f"  reg {_join_decl_tokens(_format_decl_type(port.type, {}, port.name), f'{port.name}_q')};")
    for port in scalar_outputs:
        lines.append(f"  reg {_join_decl_tokens(_format_decl_type(port.type, {}, port.name), f'{port.name}_q')};")
    for interface in memory_interfaces:
        data_type = interface.data_type
        base = interface.base
        lines.append(
            f"  reg {_join_decl_tokens(_format_decl_type(data_type, {}, base), f'{base}_mem_q [0:{_memory_depth(interface) - 1}]')};"
        )
    if scalar_inputs or scalar_outputs or memory_interfaces:
        lines.append("")

    for interface in memory_interfaces:
        base = interface.base
        lines.append(
            f"  wire {_join_decl_tokens(_format_decl_type(f'u{_memory_index_width(interface)}', {}, None), f'{base}_bus_word_addr_n')};"
        )
        lines.append(f"  wire {base}_bus_hit_n;")
    if memory_interfaces:
        lines.append("")

    for interface in memory_interfaces:
        base = interface.base
        window = next(window for window in protocol_plan.memory_windows if window.base == base)
        lines.append(
            f"  assign {base}_bus_hit_n = wb_adr_i >= {window.symbol} && "
            f"wb_adr_i < ({window.symbol} + {_format_hex32(window.span_bytes)});"
        )
        lines.append(f"  assign {base}_bus_word_addr_n = wb_adr_i[11:2];")
    if memory_interfaces:
        lines.append("")

    for interface in memory_interfaces:
        base = interface.base
        lines.append(f"  wire {_join_decl_tokens(_format_decl_type(str(interface.addr_type), {}, None), f'core_{base}_addr_n')};")
        if interface.write_type is not None:
            lines.append(f"  wire {_join_decl_tokens(_format_decl_type(str(interface.write_type), {}, None), f'core_{base}_wdata_n')};")
            lines.append(f"  wire core_{base}_we_n;")
        lines.append(f"  wire {_join_decl_tokens(_format_decl_type(str(interface.data_type), {}, None), f'core_{base}_rdata_n')};")
    for port in scalar_inputs:
        lines.append(f"  wire {_join_decl_tokens(_format_decl_type(port.type, {}, port.name), f'core_{port.name}_n')};")
    for port in scalar_outputs:
        lines.append(f"  wire {_join_decl_tokens(_format_decl_type(port.type, {}, port.name), f'core_{port.name}_n')};")
    if scalar_inputs or scalar_outputs or memory_interfaces:
        lines.append("")

    for port in scalar_inputs:
        lines.append(f"  assign core_{port.name}_n = {port.name}_q;")
    for interface in memory_interfaces:
        base = interface.base
        lines.append(f"  assign core_{base}_rdata_n = {base}_mem_q[{_memory_index_expr(f'core_{base}_addr_n', interface)}];")
    if scalar_inputs or memory_interfaces:
        lines.append("")

    lines.append("  always @(*) begin")
    lines.append("    wb_dat_o_q = 32'd0;")
    lines.append("    if (wb_req_n && !wb_we_i) begin")
    lines.append("      if (wb_adr_i == WB_REG_CONTROL_STATUS) begin")
    lines.append("        wb_dat_o_q = {28'd0, core_req_ready_n, start_pending_q, busy_q, done_q};")
    lines.append("      end")
    for port in protocol_plan.scalar_inputs:
        lines.append(f"      else if (wb_adr_i == {port.symbol}) begin")
        lines.append(f"        wb_dat_o_q = {_pack_to_wishbone(port.name + '_q', port.type)};")
        lines.append("      end")
    for port in protocol_plan.scalar_outputs:
        lines.append(f"      else if (wb_adr_i == {port.symbol}) begin")
        lines.append(f"        wb_dat_o_q = {_pack_to_wishbone(port.name + '_q', port.type)};")
        lines.append("      end")
    for interface in memory_interfaces:
        base = interface.base
        lines.append(f"      else if ({base}_bus_hit_n) begin")
        lines.append(f"        wb_dat_o_q = {_pack_to_wishbone(f'{base}_mem_q[{base}_bus_word_addr_n]', str(interface.data_type))};")
        lines.append("      end")
    lines.append("    end")
    lines.append("  end")
    lines.append("")

    lines.append("  always @(posedge clk) begin")
    lines.append("    if (rst) begin")
    lines.append("      start_pending_q <= 1'b0;")
    lines.append("      busy_q <= 1'b0;")
    lines.append("      done_q <= 1'b0;")
    for port in scalar_inputs:
        lines.append(f"      {port.name}_q <= {_zero_expr_for_type(port.type)};")
    for port in scalar_outputs:
        lines.append(f"      {port.name}_q <= {_zero_expr_for_type(port.type)};")
    lines.append("    end else begin")
    lines.append("      if (wb_req_n && wb_we_i) begin")
    lines.append("        if (wb_adr_i == WB_REG_CONTROL_STATUS && wb_dat_i[0]) begin")
    lines.append("          start_pending_q <= 1'b1;")
    lines.append("          done_q <= 1'b0;")
    lines.append("        end")
    for port in protocol_plan.scalar_inputs:
        lines.append(f"        else if (wb_adr_i == {port.symbol}) begin")
        lines.append(f"          {port.name}_q <= {_unpack_from_wishbone('wb_dat_i', port.type)};")
        lines.append("        end")
    for interface in memory_interfaces:
        if not interface.has_write:
            continue
        base = interface.base
        lines.append(f"        else if ({base}_bus_hit_n) begin")
        lines.append(f"          {base}_mem_q[{base}_bus_word_addr_n] <= {_unpack_from_wishbone('wb_dat_i', str(interface.data_type))};")
        lines.append("        end")
    lines.append("      end")
    lines.append("")
    lines.append("      if (start_pending_q && core_req_ready_n) begin")
    lines.append("        start_pending_q <= 1'b0;")
    lines.append("        busy_q <= 1'b1;")
    lines.append("      end")
    lines.append("      if (core_resp_valid_n) begin")
    lines.append("        busy_q <= 1'b0;")
    lines.append("        done_q <= 1'b1;")
    for port in scalar_outputs:
        lines.append(f"        {port.name}_q <= core_{port.name}_n;")
    lines.append("      end")
    for interface in memory_interfaces:
        if not interface.has_write:
            continue
        base = interface.base
        lines.append(f"      if (core_{base}_we_n) begin")
        lines.append(f"        {base}_mem_q[{_memory_index_expr(f'core_{base}_addr_n', interface)}] <= core_{base}_wdata_n;")
        lines.append("      end")
    lines.append("    end")
    lines.append("  end")
    lines.append("")

    lines.append(f"  {core_module_name} core (")
    core_ports = [*design.inputs, *design.outputs]
    for index, port in enumerate(core_ports):
        suffix = "," if index < len(core_ports) - 1 else ""
        signal = wrapper_core_signal(port.name, wrapper_plan)
        lines.append(f"    .{port.name}({signal}){suffix}")
    lines.append("  );")
    return lines


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
    return f"{{{32 - width}'d0, {signal}}}"


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


def _zero_expr_for_type(type_hint: str) -> str:
    if type_hint == "clock":
        return "1'b0"
    match = re.fullmatch(r"([iu])(\d+)", type_hint)
    if match is None:
        return "1'b0"
    width = max(1, int(match.group(2)))
    return _format_literal(0, width, signed=False)


def _memory_depth(interface) -> int:
    depth = getattr(interface, "depth", None)
    if isinstance(depth, int) and depth > 0:
        return depth
    return 1024


def _memory_index_width(interface) -> int:
    return max(1, ceil(log2(_memory_depth(interface))))


def _memory_index_expr(signal: str, interface) -> str:
    width = _memory_index_width(interface)
    if width == 1:
        return f"{signal}[0]"
    return f"{signal}[{width - 1}:0]"


def _sanitize_identifier(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    if not sanitized:
        sanitized = "VALUE"
    if sanitized[0].isdigit():
        sanitized = f"V_{sanitized}"
    if sanitized.lower() in _VERILOG_KEYWORDS:
        sanitized = f"{sanitized}_ID"
    return sanitized.upper()


def _join_decl_tokens(*parts: str) -> str:
    return " ".join(part for part in parts if part)
