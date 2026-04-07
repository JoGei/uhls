"""Verilog emission for uglir designs."""

from __future__ import annotations

from math import ceil, log2
import re

from uhls.backend.hls.uglir.model import UGLIRDesign, UGLIRMux, UGLIRPort, UGLIRResource, UGLIRSeqBlock, UGLIRSeqUpdate

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


def emit_uglir_to_verilog(
    design: UGLIRDesign,
    module_name: str | None = None,
    module_parameters: list[str] | None = None,
) -> str:
    """Render one uglir design as Verilog."""
    if design.stage != "uglir":
        raise ValueError(f"verilog emission expects uglir input, got stage '{design.stage}'")
    rendered_module_name = design.name if module_name is None else module_name
    if module_parameters is None:
        module_parameters = _infer_module_parameters(design)

    ctrl_widths = _ctrl_widths(design)
    ctrl_enums = _ctrl_enum_symbols(design)
    signal_types = _signal_types(design)

    lines: list[str] = []
    port_lines = [f"    {_format_port_decl(port, ctrl_widths)}" for port in [*design.inputs, *design.outputs]]
    if module_parameters:
        lines.append(f"module {rendered_module_name} #(")
        for index, parameter in enumerate(module_parameters):
            suffix = "," if index < len(module_parameters) - 1 else ""
            lines.append(f"    {parameter}{suffix}")
        lines.append(") (")
    else:
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
        if resource.kind == "mem":
            lines.append(f"  {_format_mem_decl(resource)};")
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

    for mux in design.muxes:
        lines.append(f"  assign {mux.name} = {_mux_expr(mux, ctrl_enums)};")
    if design.muxes:
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


def _infer_module_parameters(design: UGLIRDesign) -> list[str] | None:
    parameters: list[str] = []
    if any(re.search(r"\bWB_BASE_ADDR\b", str(const_decl.value)) for const_decl in design.constants):
        parameters.append("parameter [31:0] WB_BASE_ADDR = 32'h0000_0000")
    if any(re.search(r"\bOBI_BASE_ADDR\b", str(const_decl.value)) for const_decl in design.constants):
        parameters.append("parameter [31:0] OBI_BASE_ADDR = 32'h0000_0000")
    return parameters or None


def _format_port_decl(port: UGLIRPort, ctrl_widths: dict[str, int]) -> str:
    return _join_decl_tokens(port.direction, _format_decl_type(port.type, ctrl_widths, port.name), port.name)


def _signal_types(design: UGLIRDesign) -> dict[str, str]:
    signal_types: dict[str, str] = {}
    for port in [*design.inputs, *design.outputs]:
        signal_types[port.name] = port.type
    for resource in design.resources:
        if resource.kind in {"reg", "net", "mux"}:
            signal_types[resource.id] = resource.value
    return signal_types


def _ctrl_widths(design: UGLIRDesign) -> dict[str, int]:
    widths: dict[str, int] = {}
    for mux in design.muxes:
        widths[mux.select] = max(1, ceil(log2(max(len(mux.cases), 1))))
    return widths


def _ctrl_keys_for_signal(design: UGLIRDesign, select_signal: str) -> list[str]:
    for mux in design.muxes:
        if mux.select == select_signal:
            return [case.key for case in mux.cases]
    return []


def _ctrl_enum_symbols(design: UGLIRDesign) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    for mux in design.muxes:
        signal_map: dict[str, str] = {}
        for case in mux.cases:
            signal_map[case.key] = f"{_sanitize_identifier(mux.select)}_{_sanitize_identifier(case.key)}"
        mappings[mux.select] = signal_map
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


def _mux_expr(mux: UGLIRMux, ctrl_enums: dict[str, dict[str, str]]) -> str:
    symbol_map = ctrl_enums.get(mux.select, {})
    translated_sources = [_translate_expr(case.source, ctrl_enums, None) for case in mux.cases]
    if not mux.cases:
        raise ValueError(f"uglir mux '{mux.name}' must declare at least one case")
    expr = translated_sources[-1]
    for case, source in reversed(list(zip(mux.cases[:-1], translated_sources[:-1], strict=False))):
        expr = f"({mux.select} == {symbol_map[case.key]}) ? {source} : {expr}"
    return expr


def _format_seq_block(seq_block: UGLIRSeqBlock, ctrl_enums: dict[str, dict[str, str]]) -> list[str]:
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


def _format_seq_updates(updates: list[UGLIRSeqUpdate], ctrl_enums: dict[str, dict[str, str]], indent: str) -> list[str]:
    lines: list[str] = []
    for update in updates:
        if update.enable is None:
            lines.append(f"{indent}{_format_seq_update(update, ctrl_enums)}")
            continue
        lines.append(f"{indent}if ({_translate_expr(update.enable, ctrl_enums, None)}) begin")
        lines.append(f"{indent}  {_format_seq_update(UGLIRSeqUpdate(update.target, update.value), ctrl_enums)}")
        lines.append(f"{indent}end")
    return lines


def _format_seq_update(update: UGLIRSeqUpdate, ctrl_enums: dict[str, dict[str, str]]) -> str:
    return f"{update.target} <= {_translate_expr(update.value, ctrl_enums, None)};"


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


def _format_mem_decl(resource: UGLIRResource) -> str:
    type_hint, depth = _parse_mem_type(resource.value)
    return f"{_join_decl_tokens('reg', _format_decl_type(type_hint, {}, resource.id), f'{resource.id} [0:{depth - 1}]')};"


def _parse_mem_type(type_hint: str) -> tuple[str, int]:
    match = re.fullmatch(r"([iu]\d+)\[(\d+)\]", type_hint)
    if match is None:
        raise ValueError(f"unsupported uglir memory resource type '{type_hint}'; expected <scalar>[<depth>]")
    return match.group(1), int(match.group(2))


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
