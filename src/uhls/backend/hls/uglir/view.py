"""View rendering helpers for µglIR artifacts."""

from __future__ import annotations

import re

from .model import UGLIRDesign
from .pretty import format_uglir


def supported_uglir_view_values(design: UGLIRDesign) -> tuple[str, ...]:
    """List supported view names for one µglIR design."""
    return ("uglir", "mmio") if design.address_maps else ("uglir",)


def render_uglir_view(design: UGLIRDesign, *, backend: str, view_name: str) -> str:
    """Render one selected µglIR view through one backend."""
    if view_name == "uglir":
        if backend != "pretty":
            raise ValueError("µglIR view 'uglir' only supports --pretty")
        return format_uglir(design)
    if view_name == "mmio":
        if not design.address_maps:
            raise ValueError("µglIR view 'mmio' requires at least one address_map block")
        if backend == "dot":
            return format_uglir_mmio_dot(design)
        return format_uglir_mmio(design)
    raise ValueError(f"unsupported µglIR view '{view_name}'")


def format_uglir_mmio(design: UGLIRDesign) -> str:
    """Render one software-facing MMIO summary for wrapped µglIR."""
    lines: list[str] = [f"design {design.name} mmio"]
    for address_map in design.address_maps:
        lines.append(f"map {address_map.name}")
        for entry in address_map.entries:
            lines.extend(_format_mmio_entry(entry))
    return "\n".join(lines)


def format_uglir_mmio_dot(design: UGLIRDesign) -> str:
    """Render one minimal wrapped-module MMIO overview as DOT."""
    graph_name = f"{design.name}.mmio"
    address_map = design.address_maps[0] if design.address_maps else None
    address_map_name = address_map.name if address_map is not None else "mmio"
    protocol_ports = _protocol_port_names_for_address_map(address_map_name)
    memory_nodes = [
        resource for resource in design.resources if resource.kind == "mem" and resource.id.endswith("_mem_q")
    ]
    scalar_reg_nodes = [
        resource
        for resource in design.resources
        if resource.kind == "reg"
        and resource.id.endswith("_q")
        and resource.id not in {"state_q", "start_pending_q", "busy_q", "done_q", "obi_rsp_pending_q", "obi_rsp_rdata_q", "obi_rsp_head_q", "obi_rsp_tail_q", "obi_rsp_count_q"}
        and not resource.id.startswith("obi_rsp_")
        and not resource.id.startswith("wb_")
    ]
    external_port_names = [
        port.name
        for port in [*design.inputs, *design.outputs]
        if port.name not in protocol_ports and port.name not in {"clk", "rst"}
    ]
    lines = [f'digraph "{graph_name}" {{', "  rankdir=LR;", '  node [shape=box, style=rounded];']
    lines.append(f'  subgraph "cluster_wrapper" {{')
    lines.append(f'    label="wrapper {design.name}";')
    lines.append(f'    bus [label="{address_map_name.upper()} interface"];')
    if address_map is not None:
        lines.extend(_format_mmio_dot_table("mmio_map", address_map))
        lines.append('    bus -> mmio_map [label="address map"];')
    lines.append(f'    core [label="HLS core\\n{design.name}_core"];')
    if scalar_reg_nodes:
        scalar_label = "\\n".join(
            f"{resource.id.removesuffix('_q')} : {resource.value}" for resource in scalar_reg_nodes
        )
        lines.append(f'    scalars [label="scalar regs\\n{scalar_label}"];')
        lines.append("    bus -> scalars;")
        lines.append("    scalars -> core;")
        lines.append("    core -> scalars;")
    for resource in memory_nodes:
        mem_name = resource.id.removesuffix("_mem_q")
        lines.append(f'    "{resource.id}" [label="memory {mem_name}\\n{resource.value}"];')
        lines.append(f'    bus -> "{resource.id}";')
        lines.append(f'    "{resource.id}" -> core;')
        lines.append(f'    core -> "{resource.id}";')
    lines.append('    bus -> core [label="control/status"];')
    if external_port_names:
        output_label = "\\n".join(external_port_names)
        lines.append(f'    outputs [shape=note, label="external ports\\n{output_label}"];')
        lines.append("    core -> outputs;")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _format_mmio_entry(entry) -> list[str]:
    attrs = entry.attributes
    if entry.kind == "register":
        offset = attrs.get("offset", "<?>")
        access = attrs.get("access", "<?>")
        symbol = attrs.get("symbol")
        type_hint = attrs.get("type")
        suffix = f" symbol={symbol}" if symbol else ""
        if type_hint is not None:
            suffix += f" type={type_hint}"
        lines = [f"  register {entry.name} range={_format_mmio_register_range(offset, type_hint)} access={access}{suffix}"]
        if entry.name == "control_status":
            lines.extend(
                [
                    "    bit[0] done: read-only completion-latched status; write 1 starts execution and clears done",
                    "    bit[1] busy: core currently executing",
                    "    bit[2] start_pending: start command latched until req_ready",
                    "    bit[3] req_ready: core can accept a new request",
                    "    bits[31:4] reserved: read as zero",
                ]
            )
        else:
            width_desc = _bit_range_from_type(type_hint)
            data_sem = "software-visible register payload"
            if access == "rw":
                data_sem = "software-programmable scalar input"
            elif access == "ro":
                data_sem = "software-readable scalar output/status"
            lines.append(f"    {width_desc} data: {data_sem}")
        return lines
    if entry.kind == "memory":
        offset = attrs.get("offset", "<?>")
        span = attrs.get("span", "<?>")
        access = attrs.get("access", "<?>")
        word_t = attrs.get("word_t", "<?>")
        depth = attrs.get("depth", "<?>")
        symbol = attrs.get("symbol")
        suffix = f" symbol={symbol}" if symbol else ""
        return [
            f"  memory {entry.name} range={_format_mmio_range(offset, span)} access={access}{suffix}",
            f"    words: depth={depth} word_t={word_t}",
            "    addressing: word-addressed MMIO window",
        ]
    return [f"  {entry.kind} {entry.name}"]


def _bit_range_from_type(type_hint: object) -> str:
    if not isinstance(type_hint, str):
        return "bits[*]"
    match = re.fullmatch(r"[iu](\d+)", type_hint)
    if match is None:
        return "bits[*]"
    width = int(match.group(1))
    if width <= 1:
        return "bit[0]"
    return f"bits[{width - 1}:0]"


def _format_mmio_dot_table(node_name: str, address_map) -> list[str]:
    rows: list[tuple[str, str, str, str]] = []
    for entry in address_map.entries:
        if entry.kind == "register":
            offset = entry.attributes.get("offset", "<?>")
            access = entry.attributes.get("access", "<?>")
            rows.append(("reg", entry.name, _format_mmio_register_range(offset, entry.attributes.get("type")), str(access)))
        elif entry.kind == "memory":
            offset = entry.attributes.get("offset", "<?>")
            span = entry.attributes.get("span", "<?>")
            access = entry.attributes.get("access", "<?>")
            rows.append(("mem", entry.name, _format_mmio_range(offset, span), str(access)))
    html_lines = [
        f'    {node_name} [shape=plain, label=<',
        '      <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0">',
        f'        <TR><TD COLSPAN="4"><B>{_dot_html_escape(address_map.name.upper())} MMIO</B></TD></TR>',
        '        <TR><TD><B>kind</B></TD><TD><B>name</B></TD><TD><B>rel. address range</B></TD><TD><B>access</B></TD></TR>',
    ]
    for kind, name, range_text, access in rows:
        html_lines.append(
            "        <TR>"
            f"<TD>{_dot_html_escape(kind)}</TD>"
            f"<TD>{_dot_html_escape(name)}</TD>"
            f"<TD>{_dot_html_escape(range_text)}</TD>"
            f"<TD>{_dot_html_escape(access)}</TD>"
            "</TR>"
        )
    if not rows:
        html_lines.append('        <TR><TD COLSPAN="4">empty</TD></TR>')
    html_lines.extend(["      </TABLE>", '    >];'])
    return html_lines


def _dot_html_escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _format_mmio_range(offset: object, span: object) -> str:
    try:
        start = _parse_u32_like(offset)
        width = _parse_u32_like(span)
    except ValueError:
        return f"[{offset} : {offset}+{span}-1]"
    end = start + max(width - 1, 0)
    return f"[{_format_u32_hex(start)} : {_format_u32_hex(end)}]"


def _format_mmio_register_range(offset: object, type_hint: object) -> str:
    try:
        point = _parse_u32_like(offset)
        width = _type_size_bytes(type_hint)
    except ValueError:
        return f"[{offset} : {offset}]"
    return f"[{_format_u32_hex(point)} : {_format_u32_hex(point + max(width - 1, 0))}]"


def _parse_u32_like(value: object) -> int:
    text = str(value).strip().lower()
    match = re.fullmatch(r"32'h([0-9a-f]{4})_([0-9a-f]{4})", text)
    if match is None:
        raise ValueError(f"unsupported u32 literal '{value}'")
    return (int(match.group(1), 16) << 16) | int(match.group(2), 16)


def _format_u32_hex(value: int) -> str:
    return f"0x{value:x}"


def _type_size_bytes(type_hint: object) -> int:
    if not isinstance(type_hint, str):
        return 4
    match = re.fullmatch(r"[iu](\d+)", type_hint)
    if match is None:
        return 4
    width = int(match.group(1))
    return max((width + 7) // 8, 1)


def _protocol_port_names_for_address_map(address_map_name: str) -> set[str]:
    normalized = address_map_name.strip().lower()
    if normalized == "wishbone":
        return {"wb_cyc_i", "wb_stb_i", "wb_we_i", "wb_adr_i", "wb_dat_i", "wb_sel_i", "wb_dat_o", "wb_ack_o", "wb_err_o"}
    if normalized == "obi":
        return {"obi_req_i", "obi_addr_i", "obi_we_i", "obi_be_i", "obi_wdata_i", "obi_rready_i", "obi_gnt_o", "obi_rvalid_o", "obi_rdata_o"}
    return set()


__all__ = ["format_uglir_mmio", "format_uglir_mmio_dot", "render_uglir_view", "supported_uglir_view_values"]
