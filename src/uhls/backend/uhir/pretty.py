"""Pretty-print helpers for textual µhIR artifacts."""

from __future__ import annotations

from .model import (
    AttributeValue,
    UHIRConstant,
    UHIRDesign,
    UHIREdge,
    UHIRMux,
    UHIRNode,
    UHIRRegion,
    UHIRResource,
    UHIRValueBinding,
)


def format_uhir(design: UHIRDesign) -> str:
    """Render one textual µhIR design."""
    lines = [f"design {design.name}", f"stage {design.stage}"]

    for port in design.inputs:
        lines.append(f"input  {port.name} : {port.type}")
    for port in design.outputs:
        lines.append(f"output {port.name} : {port.type}")
    for const_decl in design.constants:
        lines.append(f"const  {const_decl.name} = {const_decl.value} : {const_decl.type}")

    if design.schedule is not None:
        lines.append(f"schedule kind={design.schedule.kind}")

    if design.resources:
        lines.append("resources {")
        for resource in design.resources:
            lines.append(f"  {format_resource(resource)}")
        lines.append("}")

    for region in design.regions:
        lines.append("")
        lines.extend(format_region(region))

    return "\n".join(lines)


def format_region(region: UHIRRegion) -> list[str]:
    """Render one µhIR region block."""
    attrs = [f"kind={region.kind}"]
    if region.parent is not None:
        attrs.append(f"parent={region.parent}")
    lines = [f"region {region.id} {' '.join(attrs)} {{"]

    for ref in region.region_refs:
        lines.append(f"  region_ref {ref.target}")
    for node in region.nodes:
        lines.append(f"  {format_node(node)}")
    for edge in region.edges:
        lines.append(f"  {format_edge(edge)}")
    for mapping in region.mappings:
        lines.append(f"  map {mapping.node_id} <- {mapping.source_id}")
    if region.steps is not None:
        lines.append(f"  steps [{region.steps[0]}:{region.steps[1]}]")
    if region.latency is not None:
        lines.append(f"  latency {region.latency}")
    if region.initiation_interval is not None:
        lines.append(f"  ii {region.initiation_interval}")
    for value_binding in region.value_bindings:
        lines.append(f"  {format_value_binding(value_binding)}")
    for mux in region.muxes:
        lines.append(f"  {format_mux(mux)}")

    lines.append("}")
    return lines


def format_node(node: UHIRNode) -> str:
    """Render one µhIR node."""
    expr = node.opcode if not node.operands else f"{node.opcode} {', '.join(node.operands)}"
    head = f"node {node.id} = {expr}"
    if node.result_type is not None:
        head = f"{head} : {node.result_type}"
    if node.attributes:
        head = f"{head} {_format_attrs(node.attributes)}"
    return head


def format_edge(edge: UHIREdge) -> str:
    """Render one µhIR edge."""
    edge_op = "->" if edge.directed else "--"
    head = f"edge {edge.kind} {edge.source} {edge_op} {edge.target}"
    if edge.attributes:
        head = f"{head} {_format_attrs(edge.attributes)}"
    return head


def format_value_binding(value_binding: UHIRValueBinding) -> str:
    """Render one bind-stage value binding."""
    intervals = ",".join(f"[{start}:{end}]" for start, end in value_binding.live_intervals)
    return (
        f"value {value_binding.producer} -> {value_binding.register} "
        f"live={intervals}"
    )


def format_mux(mux: UHIRMux) -> str:
    """Render one bind-stage mux declaration."""
    attrs: dict[str, AttributeValue] = dict(mux.attributes)
    attrs["input"] = mux.inputs
    attrs["output"] = mux.output
    attrs["sel"] = mux.select
    return f"mux {mux.id} : {_format_attrs(attrs)}"


def format_resource(resource: UHIRResource) -> str:
    """Render one bind-stage resource declaration."""
    if resource.kind == "fu":
        return f"fu {resource.id} : {resource.value}"
    if resource.kind == "reg":
        return f"reg {resource.id} : {resource.value}"
    if resource.kind == "port":
        suffix = "" if resource.target is None else f" {resource.target}"
        return f"port {resource.id} : {resource.value}{suffix}"
    raise TypeError(f"unsupported µhIR resource {resource!r}")


def _format_attrs(attrs: dict[str, AttributeValue]) -> str:
    return " ".join(f"{name}={_format_attr_value(value)}" for name, value in attrs.items())


def _format_attr_value(value: AttributeValue) -> str:
    if isinstance(value, tuple):
        return f"[{', '.join(value)}]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
