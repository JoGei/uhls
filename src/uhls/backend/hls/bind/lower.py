"""Operation-binding lowering entrypoint."""

from __future__ import annotations

from uhls.backend.uhir.model import (
    UHIRConstant,
    UHIRDesign,
    UHIREdge,
    UHIRMux,
    UHIRNode,
    UHIRPort,
    UHIRRegion,
    UHIRRegionRef,
    UHIRResource,
    UHIRSchedule,
    UHIRSourceMap,
    UHIRValueBinding,
)

from .interfaces import OperationBinder
from .registry import create_builtin_binder


def lower_sched_to_bind(
    design: UHIRDesign,
    *,
    binder: OperationBinder | None = None,
    algorithm: str | None = None,
) -> UHIRDesign:
    """Lower one sched-stage µhIR design to bind-stage µhIR."""
    if design.stage != "sched":
        raise ValueError(f"bind lowering expects sched-stage µhIR input, got stage '{design.stage}'")

    resolved_binder = _resolve_binder(binder=binder, algorithm=algorithm)
    binding = resolved_binder.bind_operations(design)

    bound = UHIRDesign(name=design.name, stage="bind")
    bound.inputs = [_clone_port(port) for port in design.inputs]
    bound.outputs = [_clone_port(port) for port in design.outputs]
    bound.constants = [_clone_constant(constant) for constant in design.constants]
    bound.schedule = None if design.schedule is None else UHIRSchedule(design.schedule.kind)
    bound.resources = [UHIRResource(resource.kind, resource.id, resource.value, resource.target) for resource in binding.resources]
    bound.regions = [_clone_bind_region(region, binding.node_bindings, binding.value_bindings) for region in design.regions]
    return bound


def _resolve_binder(*, binder: OperationBinder | None, algorithm: str | None) -> OperationBinder:
    if binder is not None:
        if algorithm is not None:
            raise ValueError("bind lowering accepts either one binder instance or one algorithm name, not both")
        return binder
    return create_builtin_binder("left_edge" if algorithm is None else algorithm)


def _clone_bind_region(
    region: UHIRRegion,
    node_bindings: dict[str, str],
    value_bindings: tuple[UHIRValueBinding, ...],
) -> UHIRRegion:
    cloned = UHIRRegion(id=region.id, kind=region.kind, parent=region.parent)
    cloned.region_refs = [UHIRRegionRef(ref.target) for ref in region.region_refs]
    cloned.nodes = [_clone_node(node, node_bindings) for node in region.nodes]
    cloned.edges = [UHIREdge(edge.kind, edge.source, edge.target, dict(edge.attributes), edge.directed) for edge in region.edges]
    cloned.mappings = [UHIRSourceMap(mapping.node_id, mapping.source_id) for mapping in region.mappings]
    local_value_ids = {mapping.source_id for mapping in region.mappings}
    local_value_ids.update(node.id for node in region.nodes)
    cloned.value_bindings = [
        UHIRValueBinding(binding.producer, binding.register, binding.live_intervals)
        for binding in value_bindings
        if binding.producer in local_value_ids
    ]
    cloned.muxes = [UHIRMux(mux.id, mux.inputs, mux.output, mux.select, dict(mux.attributes)) for mux in region.muxes]
    cloned.steps = region.steps
    cloned.latency = region.latency
    cloned.initiation_interval = region.initiation_interval
    return cloned


def _clone_node(node: UHIRNode, node_bindings: dict[str, str]) -> UHIRNode:
    attributes = dict(node.attributes)
    binding = node_bindings.get(node.id)
    if binding is not None:
        attributes["bind"] = binding
    return UHIRNode(node.id, node.opcode, node.operands, node.result_type, attributes)


def _clone_port(port: UHIRPort) -> UHIRPort:
    return UHIRPort(port.direction, port.name, port.type)


def _clone_constant(constant: UHIRConstant) -> UHIRConstant:
    return UHIRConstant(constant.name, constant.value, constant.type)
