"""DOT rendering helpers for canonical µhLS CFG, DFG, and CDFG views."""

from __future__ import annotations

from functools import singledispatch

from uhls.middleend.uir import (
    BinaryOp,
    CallOp,
    COMPACT_OPCODE_LABELS,
    CompareOp,
    CondBranchOp,
    Function,
    LoadOp,
    Module,
    PhiOp,
    PrintOp,
    ReturnOp,
    StoreOp,
    UnaryOp,
    format_instruction,
)
from uhls.middleend.passes.analyze import BasicBlockDFG, DFGInfo, build_cfg, build_dfg


@singledispatch
def to_dot(value: object) -> str:
    """Render one supported IR or analysis object as Graphviz DOT."""
    raise TypeError(f"cannot render DOT for {type(value).__name__}")


@to_dot.register
def _(function: Function) -> str:
    """Render one function CFG as Graphviz DOT."""
    return to_cfg_dot(function)


@to_dot.register
def _(module: Module) -> str:
    """Render one module CFG as a merged Graphviz DOT."""
    return to_module_cfg_dot(module)


@to_dot.register
def _(graph: BasicBlockDFG) -> str:
    """Render one basic-block DFG as Graphviz DOT."""
    return to_basic_block_dfg_dot(graph)


@to_dot.register
def _(info: DFGInfo) -> str:
    """Render all basic-block DFGs for one function as Graphviz DOT."""
    return to_dfg_dot(info)


def to_dfg_dot(info: DFGInfo, compact: bool = False) -> str:
    """Render all basic-block DFGs for one function as Graphviz DOT."""
    lines = [f'digraph "{info.function.name}.dfg" {{', "  compound=true;"]
    for block in info.function.blocks:
        lines.extend(
            _indent(_render_basic_block_dfg(info.blocks[block.label], standalone=False, compact=compact), "  ")
        )
    lines.append("}")
    return "\n".join(lines)


def to_basic_block_dfg_dot(graph: BasicBlockDFG, compact: bool = False) -> str:
    """Render one basic-block DFG as Graphviz DOT."""
    return "\n".join(_render_basic_block_dfg(graph, standalone=True, compact=compact))


def to_cfg_dot(function: Function) -> str:
    """Render one function CFG as Graphviz DOT."""
    cfg = build_cfg(function)
    lines = [f'digraph "{function.name}" {{', "  node [shape=box];"]

    for block in function.blocks:
        body_lines = [block.label]
        body_lines.extend(format_instruction(instruction) for instruction in block.instructions)
        if block.terminator is not None:
            body_lines.append(format_instruction(block.terminator))
        label = "\\l".join(_escape(text) for text in body_lines) + "\\l"
        lines.append(f'  "{block.label}" [label="{label}"];')

    for source in cfg.order:
        for target in sorted(cfg.successors[source]):
            lines.append(f'  "{source}":s -> "{target}":n;')

    lines.append("}")
    return "\n".join(lines)


def to_module_cfg_dot(module: Module) -> str:
    """Render all function CFGs in one merged Graphviz DOT."""
    graph_name = module.name or "module"
    lines = [f'digraph "{graph_name}.cfg" {{', "  compound=true;"]
    for function in module.functions:
        lines.extend(_indent(_render_cfg_cluster(function), "  "))
    lines.append("}")
    return "\n".join(lines)


def to_module_dfg_dot(module: Module, compact: bool = False) -> str:
    """Render all function DFGs in one merged Graphviz DOT."""
    graph_name = module.name or "module"
    lines = [f'digraph "{graph_name}.dfg" {{', "  compound=true;"]
    for function in module.functions:
        lines.extend(_indent(_render_dfg_function_cluster(function, compact=compact), "  "))
    lines.append("}")
    return "\n".join(lines)


def to_cdfg_dot(function: Function, compact: bool = False) -> str:
    """Render one combined control/data-flow graph as Graphviz DOT."""
    lines = [f'digraph "{function.name}.cdfg" {{', "  compound=true;"]
    lines.extend(_indent(_render_cdfg_contents(function, compact=compact), "  "))
    lines.append("}")
    return "\n".join(lines)


def to_module_cdfg_dot(module: Module, compact: bool = False) -> str:
    """Render all function CDFGs in one merged Graphviz DOT."""
    graph_name = module.name or "module"
    lines = [f'digraph "{graph_name}.cdfg" {{', "  compound=true;"]
    for function in module.functions:
        lines.extend(_indent(_render_cdfg_function_cluster(function, compact=compact), "  "))
    lines.append("}")
    return "\n".join(lines)


def _render_cdfg_contents(
    function: Function,
    node_prefix: str = "",
    cluster_prefix: str = "",
    compact: bool = False,
) -> list[str]:
    cfg = build_cfg(function)
    dfg = build_dfg(function)
    lines: list[str] = []

    for block in function.blocks:
        lines.extend(
            _render_basic_block_dfg(
                dfg.blocks[block.label],
                standalone=False,
                node_prefix=node_prefix,
                cluster_prefix=cluster_prefix,
                compact=compact,
            )
        )

    for source in cfg.order:
        source_exit = _block_exit_node_id(dfg.blocks[source], node_prefix=node_prefix)
        for target in sorted(cfg.successors[source]):
            target_entry = _block_entry_node_id(dfg.blocks[target], node_prefix=node_prefix)
            lines.append(
                f'"{source_exit}" -> "{target_entry}" '
                f'[color="#4c78a8", fontcolor="#4c78a8", label="ctrl", '
                f'ltail="{_dfg_cluster_id(source, cluster_prefix)}", '
                f'lhead="{_dfg_cluster_id(target, cluster_prefix)}"];'
            )
    return lines


def _render_basic_block_dfg(
    graph: BasicBlockDFG,
    standalone: bool,
    node_prefix: str = "",
    cluster_prefix: str = "",
    compact: bool = False,
) -> list[str]:
    header = (
        [f'digraph "{graph.function.name}.{graph.block}.dfg" {{', "  node [shape=ellipse];"]
        if standalone
        else [
            f'subgraph "cluster_{cluster_prefix}{graph.block}" {{',
            f'  label="{_escape(graph.block)}";',
            "  color=gray70;",
            "  node [shape=ellipse];",
        ]
    )
    lines = list(header)
    compact_inputs = _compact_input_edges(graph) if compact else {}

    if compact_inputs:
        lines.append("  {")
        lines.append("    rank=source;")
        for name in sorted(compact_inputs):
            lines.append(
                f'    "{_dfg_input_node_id(graph, name, node_prefix=node_prefix)}" '
                f'[label="{_escape(name)}", shape=box, style=filled, fillcolor="#eeeeee", color="#bdbdbd"];'
            )
        lines.append("  }")

    for node in graph.nodes:
        lines.append(
            f'  "{_dfg_node_id(graph, node.id, node_prefix=node_prefix)}" '
            f'[label="{_node_label(graph, node.id, compact=compact)}"];'
        )

    for name in sorted(compact_inputs):
        for target, kind in sorted(compact_inputs[name]):
            attrs = _edge_attributes(kind, label=name, compact=compact)
            suffix = "" if not attrs else f" [{attrs}]"
            lines.append(
                f'  "{_dfg_input_node_id(graph, name, node_prefix=node_prefix)}" -> '
                f'"{_dfg_node_id(graph, target, node_prefix=node_prefix)}"{suffix};'
            )

    for edge in graph.edges:
        attrs = _edge_attributes(edge.kind, label=edge.label, compact=compact)
        suffix = "" if not attrs else f" [{attrs}]"
        lines.append(
            f'  "{_dfg_node_id(graph, edge.source, node_prefix=node_prefix)}" -> '
            f'"{_dfg_node_id(graph, edge.target, node_prefix=node_prefix)}"{suffix};'
        )

    lines.append("}")
    return lines


def _render_dfg_function_cluster(function: Function, compact: bool = False) -> list[str]:
    dfg = build_dfg(function)
    cluster_prefix = f"{function.name}_"
    node_prefix = f"{function.name}:"
    lines = [
        f'subgraph "cluster_{function.name}_dfg" {{',
        f'  label="{_escape(function.name)}";',
        "  color=gray70;",
    ]
    for block in function.blocks:
        lines.extend(
            _indent(
                _render_basic_block_dfg(
                    dfg.blocks[block.label],
                    standalone=False,
                    node_prefix=node_prefix,
                    cluster_prefix=cluster_prefix,
                    compact=compact,
                ),
                "  ",
            )
        )
    lines.append("}")
    return lines


def _render_cdfg_function_cluster(function: Function, compact: bool = False) -> list[str]:
    node_prefix = f"{function.name}:"
    cluster_prefix = f"{function.name}_"
    lines = [
        f'subgraph "cluster_{function.name}_cdfg" {{',
        f'  label="{_escape(function.name)}";',
        "  color=gray70;",
    ]
    lines.extend(
        _indent(_render_cdfg_contents(function, node_prefix=node_prefix, cluster_prefix=cluster_prefix, compact=compact), "  ")
    )
    lines.append("}")
    return lines


def _render_cfg_cluster(function: Function) -> list[str]:
    cfg = build_cfg(function)
    lines = [
        f'subgraph "cluster_{function.name}" {{',
        f'  label="{_escape(function.name)}";',
        "  color=gray70;",
        "  node [shape=box];",
    ]

    for block in function.blocks:
        body_lines = [block.label]
        body_lines.extend(format_instruction(instruction) for instruction in block.instructions)
        if block.terminator is not None:
            body_lines.append(format_instruction(block.terminator))
        label = "\\l".join(_escape(text) for text in body_lines) + "\\l"
        lines.append(f'  "{_cfg_node_id(function.name, block.label)}" [label="{label}"];')

    for source in cfg.order:
        for target in sorted(cfg.successors[source]):
            lines.append(
                f'  "{_cfg_node_id(function.name, source)}":s -> "{_cfg_node_id(function.name, target)}":n;'
            )

    lines.append("}")
    return lines


def _node_label(graph: BasicBlockDFG, node_id: str, compact: bool = False) -> str:
    node = next(node for node in graph.nodes if node.id == node_id)
    operation = _node_operation(graph, node_id)
    if compact:
        return _escape(_compact_node_label(operation))
    return _escape(format_instruction(operation))


def _edge_attributes(kind: str, label: str = "", compact: bool = False) -> str:
    attrs: list[str] = []
    if kind == "memory":
        attrs.extend(['color="#b279a2"', 'fontcolor="#b279a2"', "style=dashed"])
    elif kind == "sink":
        attrs.extend(['color="#9c9c9c"', "style=dashed"])
    if compact and label:
        attrs.append(f'label="{_escape(label)}"')
    return ", ".join(attrs)


def _block_entry_node_id(graph: BasicBlockDFG, node_prefix: str = "") -> str:
    return _dfg_node_id(graph, graph.nodes[0].id, node_prefix=node_prefix)


def _block_exit_node_id(graph: BasicBlockDFG, node_prefix: str = "") -> str:
    return _dfg_node_id(graph, graph.nodes[-1].id, node_prefix=node_prefix)


def _indent(lines: list[str], prefix: str) -> list[str]:
    return [f"{prefix}{line}" for line in lines]


def _dfg_cluster_id(block_label: str, cluster_prefix: str = "") -> str:
    return f"cluster_{cluster_prefix}{block_label}"


def _cfg_node_id(function_name: str, block_label: str) -> str:
    return f"{function_name}:{block_label}"


def _dfg_node_id(graph: BasicBlockDFG, node_id: str, node_prefix: str = "") -> str:
    if not node_prefix:
        return node_id
    return f"{node_prefix}{node_id}"


def _dfg_input_node_id(graph: BasicBlockDFG, name: str, node_prefix: str = "") -> str:
    return _dfg_node_id(graph, f"{graph.block}:in:{name}", node_prefix=node_prefix)


def _node_operation(graph: BasicBlockDFG, node_id: str) -> object:
    node = next(node for node in graph.nodes if node.id == node_id)
    block = graph.function.block_map()[graph.block]
    if node_id.endswith(":term"):
        return block.terminator
    return block.instructions[node.index]


def _compact_input_edges(graph: BasicBlockDFG) -> dict[str, set[tuple[str, str]]]:
    incoming_labels = {(edge.target, edge.kind, edge.label) for edge in graph.edges}
    inputs: dict[str, set[tuple[str, str]]] = {}

    for node in graph.nodes:
        operation = _node_operation(graph, node.id)
        for name, kind in _operation_input_names(operation):
            if (node.id, kind, name) in incoming_labels:
                continue
            inputs.setdefault(name, set()).add((node.id, kind))

    return inputs


def _operation_input_names(operation: object) -> tuple[tuple[str, str], ...]:
    names: list[tuple[str, str]] = []

    if isinstance(operation, UnaryOp):
        if isinstance(operation.value, str):
            names.append((operation.value, "value"))
    elif isinstance(operation, (BinaryOp, CompareOp)):
        if isinstance(operation.lhs, str):
            names.append((operation.lhs, "value"))
        if isinstance(operation.rhs, str):
            names.append((operation.rhs, "value"))
    elif isinstance(operation, PhiOp):
        for incoming in operation.incoming:
            if isinstance(incoming.value, str):
                names.append((incoming.value, "value"))
    elif isinstance(operation, LoadOp):
        names.append((operation.array, "memory"))
        if isinstance(operation.index, str):
            names.append((operation.index, "value"))
    elif isinstance(operation, StoreOp):
        names.append((operation.array, "memory"))
        if isinstance(operation.index, str):
            names.append((operation.index, "value"))
        if isinstance(operation.value, str):
            names.append((operation.value, "value"))
    elif isinstance(operation, (CallOp, PrintOp)):
        for operand in operation.operands:
            if isinstance(operand, str):
                names.append((operand, "value"))
    elif isinstance(operation, CondBranchOp):
        if isinstance(operation.cond, str):
            names.append((operation.cond, "value"))
    elif isinstance(operation, ReturnOp):
        if isinstance(operation.value, str):
            names.append((operation.value, "value"))

    return tuple(names)


def _compact_node_label(operation: object) -> str:
    opcode = str(getattr(operation, "opcode", operation.__class__.__name__))
    return COMPACT_OPCODE_LABELS.get(opcode, opcode)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')
