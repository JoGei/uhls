"""Pretty-print helpers for textual µhIR artifacts."""

from __future__ import annotations

from uhls.utils.graph import topological_sort

from .model import (
    AttributeValue,
    TimingValue,
    UHIRAssign,
    UHIRAttach,
    UHIRConstant,
    UHIRController,
    UHIRControllerEmit,
    UHIRControllerLink,
    UHIRControllerState,
    UHIRControllerTransition,
    UHIRDesign,
    UHIREdge,
    UHIRGlueMux,
    UHIRGlueMuxCase,
    UHIRMux,
    UHIRNode,
    UHIRRegion,
    UHIRResource,
    UHIRSeqBlock,
    UHIRSeqUpdate,
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

    if design.stage == "uglir":
        for assign in design.assigns:
            lines.append("")
            lines.append(format_assign(assign))
        for attachment in design.attachments:
            lines.append("")
            lines.append(format_attach(attachment))
        for glue_mux in design.glue_muxes:
            lines.append("")
            lines.extend(format_glue_mux(glue_mux))
        for seq_block in design.seq_blocks:
            lines.append("")
            lines.extend(format_seq_block(seq_block))
        return "\n".join(lines)

    for controller in design.controllers:
        lines.append("")
        lines.extend(format_controller(controller))

    for region in design.regions:
        lines.append("")
        lines.extend(format_region(region))

    return "\n".join(lines)


def format_controller(controller: UHIRController) -> list[str]:
    """Render one fsm-stage controller block."""
    head = f"controller {controller.name}"
    if controller.attributes:
        head = f"{head} {_format_attrs(controller.attributes)}"
    lines = [f"{head} {{"]
    for port in controller.inputs:
        lines.append(f"  input  {port.name} : {port.type}")
    for port in controller.outputs:
        lines.append(f"  output {port.name} : {port.type}")
    for state in controller.states:
        lines.append(f"  {format_controller_state(state)}")
    for transition in controller.transitions:
        lines.append(f"  {format_controller_transition(transition)}")
    for emit in controller.emits:
        lines.append(f"  {format_controller_emit(emit)}")
    for link in controller.links:
        lines.append(f"  {format_controller_link(link)}")
    lines.append("}")
    return lines


def format_controller_state(state: UHIRControllerState) -> str:
    """Render one controller state declaration."""
    return f"state {state.name}" if not state.attributes else f"state {state.name} {_format_attrs(state.attributes)}"


def format_controller_transition(transition: UHIRControllerTransition) -> str:
    """Render one controller transition declaration."""
    head = f"transition {transition.source} -> {transition.target}"
    return head if not transition.attributes else f"{head} {_format_attrs(transition.attributes)}"


def format_controller_emit(emit: UHIRControllerEmit) -> str:
    """Render one controller emit declaration."""
    head = f"emit {emit.state}"
    return head if not emit.attributes else f"{head} {_format_attrs(emit.attributes)}"


def format_controller_link(link: UHIRControllerLink) -> str:
    """Render one controller link declaration."""
    head = f"link {link.child} via={link.node}"
    return head if not link.attributes else f"{head} {_format_attrs(link.attributes)}"


def format_assign(assign: UHIRAssign) -> str:
    """Render one uglir combinational assignment."""
    return f"assign {assign.target} = {assign.expr}"


def format_attach(attachment: UHIRAttach) -> str:
    """Render one uglir instance-port attachment."""
    return f"{attachment.instance}.{attachment.port}({attachment.signal})"


def format_glue_mux(glue_mux: UHIRGlueMux) -> list[str]:
    """Render one uglir mux declaration."""
    lines = [f"mux {glue_mux.name} : {glue_mux.type} sel={glue_mux.select} {{"]
    for case in glue_mux.cases:
        lines.append(f"  {format_glue_mux_case(case)}")
    lines.append("}")
    return lines


def format_glue_mux_case(case: UHIRGlueMuxCase) -> str:
    """Render one uglir mux case."""
    return f"{case.key} -> {case.source}"


def format_seq_block(seq_block: UHIRSeqBlock) -> list[str]:
    """Render one uglir sequential block."""
    lines = [f"seq {seq_block.clock} {{"]
    if seq_block.reset is not None:
        lines.append(f"  if {seq_block.reset} {{")
        for update in seq_block.reset_updates:
            lines.append(f"    {format_seq_update(update)}")
        lines.append("  } else {")
        for update in seq_block.updates:
            if update.enable is None:
                lines.append(f"    {format_seq_update(update)}")
            else:
                lines.append(f"    if {update.enable} {{")
                lines.append(f"      {format_seq_update(UHIRSeqUpdate(update.target, update.value))}")
                lines.append("    }")
        lines.append("  }")
    else:
        for update in seq_block.updates:
            if update.enable is None:
                lines.append(f"  {format_seq_update(update)}")
            else:
                lines.append(f"  if {update.enable} {{")
                lines.append(f"    {format_seq_update(UHIRSeqUpdate(update.target, update.value))}")
                lines.append("  }")
    lines.append("}")
    return lines


def format_seq_update(update: UHIRSeqUpdate) -> str:
    """Render one uglir sequential update."""
    return f"{update.target} <= {update.value}"


def format_region(region: UHIRRegion) -> list[str]:
    """Render one µhIR region block."""
    attrs = [f"kind={region.kind}"]
    if region.parent is not None:
        attrs.append(f"parent={region.parent}")
    lines = [f"region {region.id} {' '.join(attrs)} {{"]

    ordered_refs = sorted(region.region_refs, key=lambda ref: ref.target)
    ordered_nodes = _order_region_nodes(region)
    node_position = {node.id: index for index, node in enumerate(ordered_nodes)}
    ordered_edges = _order_region_edges(region, node_position)
    ordered_mappings = sorted(
        region.mappings,
        key=lambda mapping: (node_position.get(mapping.node_id, len(node_position)), mapping.node_id, mapping.source_id),
    )
    ordered_value_bindings = sorted(
        region.value_bindings,
        key=lambda binding: (binding.producer, binding.register, binding.live_intervals),
    )
    ordered_muxes = sorted(region.muxes, key=lambda mux: mux.id)

    for ref in ordered_refs:
        lines.append(f"  region_ref {ref.target}")
    for node in ordered_nodes:
        lines.append(f"  {format_node(node)}")
    for edge in ordered_edges:
        lines.append(f"  {format_edge(edge)}")
    for mapping in ordered_mappings:
        lines.append(f"  map {mapping.node_id} <- {mapping.source_id}")
    if region.steps is not None:
        lines.append(f"  steps [{_format_timing_value(region.steps[0])}:{_format_timing_value(region.steps[1])}]")
    if region.latency is not None:
        lines.append(f"  latency {_format_timing_value(region.latency)}")
    if region.initiation_interval is not None:
        lines.append(f"  ii {_format_timing_value(region.initiation_interval)}")
    for value_binding in ordered_value_bindings:
        lines.append(f"  {format_value_binding(value_binding)}")
    for mux in ordered_muxes:
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
    if resource.kind == "net":
        return f"net {resource.id} : {resource.value}"
    if resource.kind == "inst":
        return f"inst {resource.id} : {resource.value}"
    if resource.kind == "mux":
        return f"mux {resource.id} : {resource.value}"
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


def _format_timing_value(value: TimingValue) -> str:
    return str(value)


def _order_region_nodes(region: UHIRRegion) -> list[UHIRNode]:
    base_nodes = sorted(region.nodes, key=_node_priority)
    node_by_id = {node.id: node for node in base_nodes}

    def neighbors(node: UHIRNode) -> tuple[UHIRNode, ...]:
        next_nodes: list[UHIRNode] = []
        for edge in region.edges:
            if not edge.directed:
                continue
            if edge.source != node.id or edge.target not in node_by_id:
                continue
            if edge.kind == "seq" and edge.target not in node_by_id:
                continue
            target = node_by_id.get(edge.target)
            if target is not None:
                next_nodes.append(target)
        return tuple(next_nodes)

    try:
        return topological_sort(base_nodes, neighbors, key=lambda node: node.id)
    except ValueError:
        return base_nodes


def _node_priority(node: UHIRNode) -> tuple[int, str, str]:
    role = node.attributes.get("role")
    if node.opcode == "nop" and role == "source":
        return (0, node.id, node.opcode)
    if node.opcode == "phi":
        return (1, node.id, node.opcode)
    if node.opcode == "nop" and role == "sink":
        return (4, node.id, node.opcode)
    return (2, node.id, node.opcode)


def _order_region_edges(region: UHIRRegion, node_position: dict[str, int]) -> list[UHIREdge]:
    kind_rank = {"data": 0, "mem": 1, "seq": 2}

    def endpoint_rank(endpoint: str) -> tuple[int, object]:
        if endpoint in node_position:
            return (0, node_position[endpoint])
        return (1, endpoint)

    return sorted(
        region.edges,
        key=lambda edge: (
            kind_rank.get(edge.kind, 9),
            endpoint_rank(edge.source),
            endpoint_rank(edge.target),
            0 if edge.directed else 1,
            tuple(sorted(edge.attributes.items())),
        ),
    )
