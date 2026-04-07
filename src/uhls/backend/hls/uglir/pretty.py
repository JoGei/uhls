"""Pretty-print helpers for textual µglIR artifacts."""

from __future__ import annotations

import textwrap

from .model import (
    UGLIRAddressMap,
    UGLIRAddressMapEntry,
    UGLIRAssign,
    UGLIRAttach,
    UGLIRConstant,
    UGLIRDesign,
    UGLIRMux,
    UGLIRMuxCase,
    UGLIRResource,
    UGLIRSeqBlock,
    UGLIRSeqUpdate,
)


def format_uglir(design: UGLIRDesign) -> str:
    """Render one textual µglIR design."""
    if design.stage != "uglir":
        raise ValueError(f"µglIR formatter expects stage 'uglir', got stage '{design.stage}'")

    lines = [f"design {design.name}", "stage uglir"]
    for port in design.inputs:
        lines.append(f"input  {port.name} : {port.type}")
    for port in design.outputs:
        lines.append(f"output {port.name} : {port.type}")
    for const_decl in design.constants:
        lines.extend(format_constant(const_decl))
    for address_map in design.address_maps:
        lines.append("")
        lines.extend(format_address_map(address_map))
    if design.resources:
        lines.append("resources {")
        for resource in design.resources:
            lines.append(f"  {format_resource(resource)}")
        lines.append("}")
    for assign in design.assigns:
        lines.append("")
        lines.extend(format_assign(assign))
    for attachment in design.attachments:
        lines.append("")
        lines.append(format_attach(attachment))
    for mux in design.muxes:
        lines.append("")
        lines.extend(format_mux(mux))
    for seq_block in design.seq_blocks:
        lines.append("")
        lines.extend(format_seq_block(seq_block))
    return "\n".join(lines)


def format_constant(const_decl: UGLIRConstant) -> list[str]:
    """Render one top-level constant declaration."""
    return _format_expr_statement(f"const  {const_decl.name} =", str(const_decl.value), suffix=f" : {const_decl.type}")


def format_address_map(address_map: UGLIRAddressMap) -> list[str]:
    """Render one top-level address map declaration."""
    lines = [f"address_map {address_map.name} {{"]
    for entry in address_map.entries:
        lines.append(f"  {format_address_map_entry(entry)}")
    lines.append("}")
    return lines


def format_address_map_entry(entry: UGLIRAddressMapEntry) -> str:
    """Render one address-map entry."""
    head = f"{entry.kind} {entry.name}"
    return head if not entry.attributes else f"{head} {_format_attrs(entry.attributes)}"


def format_resource(resource: UGLIRResource) -> str:
    """Render one µglIR resource declaration."""
    if resource.kind == "reg":
        return f"reg {resource.id} : {resource.value}"
    if resource.kind == "net":
        return f"net {resource.id} : {resource.value}"
    if resource.kind == "mem":
        return f"mem {resource.id} : {resource.value}"
    if resource.kind == "inst":
        return f"inst {resource.id} : {resource.value}"
    if resource.kind == "mux":
        return f"mux {resource.id} : {resource.value}"
    if resource.kind == "port":
        suffix = "" if resource.target is None else f" {resource.target}"
        return f"port {resource.id} : {resource.value}{suffix}"
    raise TypeError(f"unsupported µglIR resource {resource!r}")


def format_assign(assign: UGLIRAssign) -> list[str]:
    """Render one combinational assignment."""
    return _format_expr_statement(f"assign {assign.target} =", assign.expr)


def format_attach(attachment: UGLIRAttach) -> str:
    """Render one instance-port attachment."""
    return f"{attachment.instance}.{attachment.port}({attachment.signal})"


def format_mux(mux: UGLIRMux) -> list[str]:
    """Render one explicit mux declaration."""
    lines = [f"mux {mux.name} : {mux.type} sel={mux.select} {{"]
    for case in mux.cases:
        lines.append(f"  {format_mux_case(case)}")
    lines.append("}")
    return lines


def format_mux_case(case: UGLIRMuxCase) -> str:
    """Render one mux case."""
    return f"{case.key} -> {case.source}"


def format_seq_block(seq_block: UGLIRSeqBlock) -> list[str]:
    """Render one sequential block."""
    lines = [f"seq {seq_block.clock} {{"]
    if seq_block.reset is not None:
        lines.extend(_indent_lines(_format_if_header(seq_block.reset), "  "))
        for update in seq_block.reset_updates:
            lines.extend(_indent_lines(format_seq_update(update), "    "))
        lines.append("  } else {")
        for update in seq_block.updates:
            if update.enable is None:
                lines.extend(_indent_lines(format_seq_update(update), "    "))
            else:
                lines.extend(_indent_lines(_format_if_header(update.enable), "    "))
                lines.extend(_indent_lines(format_seq_update(UGLIRSeqUpdate(update.target, update.value)), "      "))
                lines.append("    }")
        lines.append("  }")
    else:
        for update in seq_block.updates:
            if update.enable is None:
                lines.extend(_indent_lines(format_seq_update(update), "  "))
            else:
                lines.extend(_indent_lines(_format_if_header(update.enable), "  "))
                lines.extend(_indent_lines(format_seq_update(UGLIRSeqUpdate(update.target, update.value)), "    "))
                lines.append("  }")
    lines.append("}")
    return lines


def format_seq_update(update: UGLIRSeqUpdate) -> list[str]:
    """Render one sequential update."""
    return _format_expr_statement(f"{update.target} <=", update.value)


def _format_attrs(attrs: dict[str, str | int | bool | tuple[str, ...]]) -> str:
    return " ".join(f"{name}={_format_attr_value(value)}" for name, value in attrs.items())


def _format_attr_value(value: str | int | bool | tuple[str, ...]) -> str:
    if isinstance(value, tuple):
        return f"[{', '.join(value)}]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_if_header(condition: str) -> list[str]:
    return _format_expr_statement("if", condition, suffix=" {")


def _format_expr_statement(prefix: str, expr: str, *, suffix: str = "", width: int = 100) -> list[str]:
    single = f"{prefix} {expr}{suffix}"
    if len(single) <= width:
        return [single]
    wrapped = textwrap.wrap(
        expr,
        width=max(40, width - 4),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(wrapped) <= 1:
        return [single]
    lines = [f"{prefix} ("]
    lines.extend(f"  {part}" for part in wrapped)
    lines.append(f"){suffix}")
    return lines


def _indent_lines(lines: list[str], indent: str) -> list[str]:
    return [f"{indent}{line}" for line in lines]


__all__ = ["format_uglir"]
