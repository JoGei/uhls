"""Operation-binding lowering entrypoint."""

from __future__ import annotations

import re

from uhls.backend.hls.uhir.model import (
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
    bound.resources.extend(_infer_memory_port_resources(design, binding.resources, binding.node_bindings))
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


def _infer_memory_port_resources(
    design: UHIRDesign,
    binding_resources: tuple[UHIRResource, ...],
    node_bindings: dict[str, str],
) -> list[UHIRResource]:
    memref_types = {
        port.name: port.type
        for port in [*design.inputs, *design.outputs]
        if port.type.startswith("memref<")
    }
    if not memref_types:
        return []

    resource_type_by_id = {
        resource.id: resource.value
        for resource in binding_resources
        if resource.kind == "fu"
    }
    inferred: dict[str, str] = {}
    for region in design.regions:
        for node in region.nodes:
            if node.opcode not in {"load", "store"} or not node.operands:
                continue
            memory_name = node.operands[0]
            memref_type = memref_types.get(memory_name)
            if memref_type is None:
                continue
            resource_id = node_bindings.get(node.id)
            if resource_id is None:
                continue
            component_name = resource_type_by_id.get(resource_id)
            if component_name is None:
                continue
            component_name = _parameterized_memory_component_type(component_name, memref_type)
            existing = inferred.get(memory_name)
            if existing is not None and existing != component_name:
                raise ValueError(
                    f"bind lowering inferred inconsistent memory port component types for '{memory_name}': "
                    f"'{existing}' vs '{component_name}'"
                )
            inferred[memory_name] = component_name

    existing_ports = {
        resource.id
        for resource in binding_resources
        if resource.kind == "port"
    }
    return [
        UHIRResource("port", memory_name, component_name, memory_name)
        for memory_name, component_name in sorted(inferred.items())
        if memory_name not in existing_ports
    ]


def _parameterized_memory_component_type(component_name: str, memref_type: str) -> str:
    element_type, extent = _parse_memref_type(memref_type)
    params = [f"word_t={element_type}"]
    if extent is not None:
        params.append(f"word_len={extent}")
    return f"{component_name}<{','.join(params)}>"


def _parse_memref_type(type_name: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"memref<\s*([A-Za-z_][\w$<>]*)\s*(?:,\s*(\d+)\s*)?>", type_name)
    if match is None:
        raise ValueError(f"invalid memref type '{type_name}'")
    element_type, extent_text = match.groups()
    return element_type, None if extent_text is None else int(extent_text)
