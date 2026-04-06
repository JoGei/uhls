"""DOT rendering helpers for µhIR sequencing graphs."""

from __future__ import annotations

from uhls.middleend.uir import COMPACT_OPCODE_LABELS
from uhls.utils.dot import escape_dot_label as _escape
from uhls.utils.dot import indent_lines as _indent

from .model import UHIRDesign, UHIRRegion


def to_dot(design: UHIRDesign, compact: bool = False) -> str:
    """Render one µhIR design as Graphviz DOT."""
    lines = [f'digraph "{design.name}.{design.stage}" {{', "  compound=true;"]
    for region in design.regions:
        lines.extend(_indent(_render_region(region, compact=compact), "  "))
    for region in design.regions:
        lines.extend(_indent(_render_region_edges(region, design), "  "))
    lines.append("}")
    return "\n".join(lines)


def _render_region(region: UHIRRegion, compact: bool = False) -> list[str]:
    lines = [f'subgraph "cluster_{region.id}" {{', f'  label="{_escape(region.id)} ({_escape(region.kind)})";']
    lines.append("  color=gray70;")
    for node in region.nodes:
        label = _escape(_node_label(node, compact=compact))
        is_hierarchical = node.opcode in {"call", "loop", "branch", "sel"}
        shape = "box" if is_hierarchical else "ellipse"
        fillcolor = "#e6e6e6" if is_hierarchical else "#e8eef8" if node.opcode == "nop" else "#ffffff"
        lines.append(
            f'  "{node.id}" [label="{label}", shape={shape}, style=filled, fillcolor="{fillcolor}"];'
        )
    lines.append("}")
    return lines


def _render_region_edges(region: UHIRRegion, design: UHIRDesign) -> list[str]:
    lines: list[str] = []
    for edge in region.edges:
        source = _edge_endpoint(edge.source, design, sink=True)
        target = _edge_endpoint(edge.target, design, sink=False)
        attrs = _dot_edge_attrs(
            edge.kind,
            edge.attributes.get("when"),
            bool(edge.attributes.get("hierarchy")),
            directed=edge.directed,
        )
        lines.append(f'"{source}" -> "{target}"{attrs};')
    return lines


def _edge_endpoint(endpoint: str, design: UHIRDesign, sink: bool) -> str:
    region = design.get_region(endpoint)
    if region is None:
        return endpoint
    boundary = _region_boundary_node(region, "sink" if sink else "source")
    return boundary.id if boundary is not None else endpoint


def _region_boundary_node(region: UHIRRegion, role: str) -> object | None:
    for node in region.nodes:
        if node.opcode == "nop" and node.attributes.get("role") == role:
            return node
    return None


def _dot_edge_attrs(kind: str, branch_condition: object, hierarchy: bool, *, directed: bool = True) -> str:
    attrs: list[str] = []
    if hierarchy:
        attrs.extend(['color="#4c78a8"', 'fontcolor="#4c78a8"', 'style=dashed'])
    elif kind == "seq":
        attrs.extend(['color="#4c78a8"', 'fontcolor="#4c78a8"'])
    elif kind == "data":
        attrs.extend(['color="#222222"'])
    elif kind == "mem":
        attrs.extend(['color="#9c755f"', 'fontcolor="#9c755f"'])
    else:
        attrs.extend(['color="#666666"', 'style=dashed'])
    if branch_condition is True:
        attrs.append('label="true"')
    elif branch_condition is False:
        attrs.append('label="false"')
    if not directed:
        attrs.append('dir=none')
    return "" if not attrs else f" [{', '.join(attrs)}]"
def _node_label(node: object, compact: bool = False) -> str:
    if compact:
        return _compact_node_label(node)
    head = f"{node.id}: {node.opcode}"
    operands = getattr(node, "operands", ())
    if operands:
        head = f"{head} {', '.join(str(item) for item in operands)}"
    result_type = getattr(node, "result_type", None)
    if result_type is not None:
        head = f"{head} : {result_type}"
    role = getattr(node, "attributes", {}).get("role")
    if role is not None and node.opcode == "nop":
        head = f"{head} ({role})"
    return head


def _compact_node_label(node: object) -> str:
    opcode = getattr(node, "opcode", "")
    structural = {"call", "loop", "branch", "nop", "sel"}
    if opcode in structural:
        short = opcode
    else:
        short = COMPACT_OPCODE_LABELS.get(opcode, opcode)
    return f"{node.id}: {short}"
