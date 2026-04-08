"""Resource-type allocation for seq-stage µhIR."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re

from uhls.backend.hls.impl import MemoryPolicy, select_memory_implementation
from uhls.backend.hls.lib import format_component_spec
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
from uhls.middleend.uir import COMPACT_OPCODE_LABELS
from uhls.utils.dot import escape_dot_label

_UIR_LANGUAGE_OPCODES = frozenset(COMPACT_OPCODE_LABELS)
_CONTROL_FU = "CTRL"
_FIXED_ALLOCATIONS = {
    "NOP": (_CONTROL_FU, 0, 0),
    "nop": (_CONTROL_FU, 0, 0),
    "BRANCH": (_CONTROL_FU, 0, 0),
    "branch": (_CONTROL_FU, 0, 0),
    "LOOP": (_CONTROL_FU, 0, 0),
    "loop": (_CONTROL_FU, 0, 0),
    "CALL": (_CONTROL_FU, 0, 0),
    "call": (_CONTROL_FU, 0, 0),
    "PHI": (_CONTROL_FU, 0, 0),
    "phi": (_CONTROL_FU, 0, 0),
    "PRINT": (_CONTROL_FU, 0, 0),
    "print": (_CONTROL_FU, 0, 0),
    "CONST": (_CONTROL_FU, 0, 0),
    "const": (_CONTROL_FU, 0, 0),
    "PARAM": (_CONTROL_FU, 0, 0),
    "param": (_CONTROL_FU, 0, 0),
    "sel": (_CONTROL_FU, 0, 0),
    "ret": (_CONTROL_FU, 0, 0),
}
_ALLOCATION_ALGORITHMS = frozenset({"min_delay", "min_ii"})
_STRUCTURAL_EXECUTABILITY_OPS = frozenset({"nop", "branch", "loop", "call", "phi", "print", "const", "param", "sel", "ret"})
_IMPLICIT_EXECUTABILITY_OPS = frozenset({"br", "cbr", "call", "const", "param", "phi", "print", "ret"})


def dummy_executability_graph() -> ExecutabilityGraph:
    """Return one starter executability graph covering all canonical µIR ops."""
    operations = tuple(sorted(_UIR_LANGUAGE_OPCODES))
    return ExecutabilityGraph(
        functional_units=("EWMS",),
        operations=operations,
        edges=tuple(("EWMS", operation, 1, 1) for operation in operations),
        )


@dataclass(slots=True, frozen=True)
class ExecutabilityGraph:
    """One user-provided weighted bipartite executability graph."""

    functional_units: tuple[str, ...]
    operations: tuple[str, ...]
    edges: tuple[tuple[str, str, int, int], ...]
    support_types: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = ()

    def __post_init__(self) -> None:
        normalized_fus = tuple(dict.fromkeys(_normalize_fu_name(fu) for fu in self.functional_units))
        normalized_ops = tuple(dict.fromkeys(_normalize_operation_name(operation) for operation in self.operations))
        fu_set = {str(fu) for fu in self.functional_units} | set(normalized_fus)
        op_set = {str(operation) for operation in self.operations} | set(normalized_ops)
        normalized_edges: list[tuple[str, str, int, int]] = []
        for source, target, ii, delay in self.edges:
            source_name = str(source)
            target_name = str(target)
            if source_name in fu_set:
                normalized_edges.append((_normalize_fu_name(source_name), _normalize_operation_name(target_name), ii, delay))
            elif source_name in op_set:
                normalized_edges.append((_normalize_operation_name(source_name), _normalize_fu_name(target_name), ii, delay))
            else:
                normalized_edges.append((source_name, target_name, ii, delay))
        normalized_support_types: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []
        for functional_unit, operation, bindings in self.support_types:
            normalized_support_types.append(
                (
                    _normalize_fu_name(functional_unit),
                    _normalize_operation_name(operation),
                    tuple((str(binding_name), str(binding_type)) for binding_name, binding_type in bindings),
                )
            )
        object.__setattr__(self, "functional_units", normalized_fus)
        object.__setattr__(self, "operations", normalized_ops)
        object.__setattr__(self, "edges", tuple(normalized_edges))
        object.__setattr__(self, "support_types", tuple(normalized_support_types))

    @classmethod
    def from_mapping(
        cls,
        graph: Mapping[str, Mapping[str, int | Mapping[str, int]] | Iterable[str]],
    ) -> ExecutabilityGraph:
        """Build one executability graph from FU adjacency."""
        functional_units = tuple(str(fu) for fu in graph)
        operation_order: dict[str, None] = {}
        normalized_edges: list[tuple[str, str, int, int]] = []
        for fu, supported_operations in graph.items():
            if isinstance(supported_operations, Mapping):
                items = supported_operations.items()
            else:
                items = ((operation, {"ii": 1, "d": 1}) for operation in supported_operations)
            for operation, weight in items:
                operation_name = str(operation)
                operation_order.setdefault(operation_name, None)
                ii, delay = _normalize_weight(weight)
                normalized_edges.append((str(fu), operation_name, ii, delay))
        return cls(
            functional_units=functional_units,
            operations=tuple(operation_order),
            edges=tuple(normalized_edges),
        )


@dataclass(slots=True, frozen=True)
class AllocationCandidate:
    """One typed allocation candidate for one opcode."""

    functional_unit: str
    ii: int
    delay: int
    type_constraints: tuple[tuple[str, str], ...] = ()


def executability_graph_from_uhir(design: UHIRDesign) -> ExecutabilityGraph:
    """Build one executability graph from one exg-stage µhIR design."""
    if design.stage != "exg":
        raise ValueError(f"executability µhIR input must use stage 'exg', got stage '{design.stage}'")

    functional_units: list[str] = []
    operations: list[str] = []
    edges: list[tuple[str, str, int, int]] = []
    vertex_names: dict[str, str] = {}
    for region in design.regions:
        for node in region.nodes:
            partition = node.attributes.get("partition")
            vertex_name = node.attributes.get("name")
            resolved_name = vertex_name if isinstance(vertex_name, str) and vertex_name else node.id
            vertex_names[node.id] = resolved_name
            if partition == "fu":
                functional_units.append(resolved_name)
            elif partition == "op":
                operations.append(resolved_name)
            else:
                raise ValueError(f"exg-stage node '{node.id}' must declare partition=fu|op")
        for edge in region.edges:
            ii = edge.attributes.get("ii")
            delay = edge.attributes.get("d")
            if not isinstance(ii, int) or not isinstance(delay, int):
                raise ValueError(f"exg-stage edge '{edge.source} -- {edge.target}' must define integer ii/d weights")
            source_name = vertex_names.get(edge.source)
            target_name = vertex_names.get(edge.target)
            if source_name is None or target_name is None:
                raise ValueError(f"exg-stage edge '{edge.source} -- {edge.target}' references unknown vertices")
            edges.append((source_name, target_name, ii, delay))

    return ExecutabilityGraph(
        functional_units=tuple(functional_units),
        operations=tuple(operations),
        edges=tuple(edges),
    )


def lower_seq_to_alloc(
    design: UHIRDesign,
    *,
    executability_graph: UHIRDesign | ExecutabilityGraph | Mapping[str, Mapping[str, int] | Iterable[str]],
    algorithm: str = "min_delay",
    memory_policy: MemoryPolicy | None = None,
    memory_vendor_components: Mapping[str, Mapping[str, object]] | None = None,
) -> UHIRDesign:
    """Annotate one seq-stage design with chosen resource types and delays."""
    if design.stage != "seq":
        raise ValueError(f"alloc lowering expects seq-stage µhIR input, got stage '{design.stage}'")

    graph = _normalize_executability_graph(executability_graph)
    candidates = _validate_and_collect_allocations(graph, algorithm=algorithm)
    value_types = _design_value_types(design)
    memory_impls = _design_memory_implementations(
        design,
        memory_policy=memory_policy,
        memory_vendor_components=memory_vendor_components,
    )

    allocated = UHIRDesign(name=design.name, stage="alloc")
    allocated.inputs = [_clone_port(port) for port in design.inputs]
    allocated.outputs = [_clone_port(port) for port in design.outputs]
    allocated.constants = [_clone_constant(constant) for constant in design.constants]
    allocated.schedule = _clone_schedule(design.schedule)
    allocated.resources = [_clone_resource(resource) for resource in design.resources]
    allocated.regions = [
        _allocate_region(region, candidates, value_types, memory_impls, algorithm=algorithm) for region in design.regions
    ]
    allocated.regions.extend(
        _embed_executability_regions(
            graph,
            {region.id for region in allocated.regions},
            {node.id for region in allocated.regions for node in region.nodes},
            _used_executability_operations(allocated.regions),
        )
    )
    return allocated


def format_executability_graph(
    graph: UHIRDesign | ExecutabilityGraph | Mapping[str, Mapping[str, int | Mapping[str, int]] | Iterable[str]],
) -> str:
    """Render one executability graph in a compact text form."""
    normalized = _normalize_executability_graph(graph)
    lines = ["executability_graph {"]
    for fu in sorted(normalized.functional_units):
        lines.append(f"  fu {fu}")
    for operation in sorted(normalized.operations):
        lines.append(f"  op {operation}")
    for source, target, ii, delay in sorted(normalized.edges, key=lambda item: (item[0], item[1], item[2], item[3])):
        lines.append(f"  edge {source} -- {target} ii={ii} d={delay}")
    lines.append("}")
    return "\n".join(lines)


def executability_graph_to_dot(
    graph: UHIRDesign | ExecutabilityGraph | Mapping[str, Mapping[str, int | Mapping[str, int]] | Iterable[str]],
) -> str:
    """Render one executability graph as Graphviz DOT."""
    normalized = _normalize_executability_graph(graph)
    lines = ['graph "executability_graph" {', "  rankdir=LR;"]
    lines.append('  subgraph "cluster_fu" {')
    lines.append('    label="functional_units";')
    for fu in sorted(normalized.functional_units):
        lines.append(f'    "fu:{fu}" [label="{escape_dot_label(fu)}", shape=box, style=filled, fillcolor="#e8eef8"];')
    lines.append("  }")
    lines.append('  subgraph "cluster_op" {')
    lines.append('    label="operations";')
    for operation in sorted(normalized.operations):
        lines.append(
            f'    "op:{operation}" [label="{escape_dot_label(operation)}", shape=ellipse, style=filled, fillcolor="#ffffff"];'
        )
    lines.append("  }")
    for source, target, ii, delay in sorted(normalized.edges, key=lambda item: (item[0], item[1], item[2], item[3])):
        if source in normalized.functional_units:
            fu = source
            operation = target
        else:
            fu = target
            operation = source
        lines.append(
            f'  "fu:{fu}" -- "op:{operation}" [label="ii={ii}, d={delay}", color="#4c78a8", fontcolor="#4c78a8"];'
        )
    lines.append("}")
    return "\n".join(lines)


def _normalize_executability_graph(
    graph: UHIRDesign | ExecutabilityGraph | Mapping[str, Mapping[str, int | Mapping[str, int]] | Iterable[str]],
) -> ExecutabilityGraph:
    if isinstance(graph, UHIRDesign):
        return executability_graph_from_uhir(graph)
    if isinstance(graph, ExecutabilityGraph):
        return graph
    return ExecutabilityGraph.from_mapping(graph)


def _validate_and_collect_allocations(
    graph: ExecutabilityGraph,
    *,
    algorithm: str,
) -> dict[str, list[AllocationCandidate]]:
    if algorithm not in _ALLOCATION_ALGORITHMS:
        supported = ", ".join(sorted(_ALLOCATION_ALGORITHMS))
        raise ValueError(f"unsupported allocation algorithm '{algorithm}'; expected one of: {supported}")

    functional_units = {vertex for vertex in graph.functional_units}
    operations = {vertex for vertex in graph.operations}
    overlap = functional_units & operations
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"executability graph is not bipartite: shared FU/op vertices: {names}")

    support_type_map = {
        (functional_unit, operation): dict(bindings)
        for functional_unit, operation, bindings in graph.support_types
    }
    executable_on: dict[str, list[AllocationCandidate]] = {operation: [] for operation in operations}
    for source, target, ii, delay in graph.edges:
        if ii <= 0:
            raise ValueError(f"executability graph edge {source!r} -> {target!r} has non-positive ii {ii}")
        if delay < 0:
            raise ValueError(f"executability graph edge {source!r} -> {target!r} has negative delay {delay}")
        if ii > delay:
            raise ValueError(f"executability graph edge {source!r} -> {target!r} violates ii<=d: ii={ii}, d={delay}")
        if source in functional_units and target in operations:
            executable_on[target].append(
                AllocationCandidate(source, ii, delay, tuple(sorted(support_type_map.get((source, target), {}).items())))
            )
            continue
        if source in operations and target in functional_units:
            executable_on[source].append(
                AllocationCandidate(target, ii, delay, tuple(sorted(support_type_map.get((target, source), {}).items())))
            )
            continue
        if source in functional_units or target in functional_units or source in operations or target in operations:
            raise ValueError(
                f"executability graph is not bipartite: edge {source!r} -> {target!r} must connect one FU and one op"
            )
        raise ValueError(f"executability graph edge {source!r} -> {target!r} references unknown vertices")

    missing = sorted(
        operation
        for operation in _UIR_LANGUAGE_OPCODES
        if operation not in _IMPLICIT_EXECUTABILITY_OPS and not executable_on.get(operation)
    )
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"executability graph does not cover all canonical µIR operations: {names}")

    return executable_on


def _choose_allocation_candidate(
    candidates: list[AllocationCandidate],
    *,
    algorithm: str,
) -> AllocationCandidate:
    if algorithm == "min_delay":
        return min(candidates, key=lambda item: (item.delay, item.ii, item.functional_unit))
    if algorithm == "min_ii":
        return min(candidates, key=lambda item: (item.ii, item.delay, item.functional_unit))
    raise AssertionError(f"unreachable allocation algorithm '{algorithm}'")


def _embed_executability_regions(
    graph: ExecutabilityGraph,
    taken_region_ids: set[str],
    taken_node_ids: set[str],
    used_operations: set[str],
) -> list[UHIRRegion]:
    retained_edges = [
        (fu, operation, ii, delay)
        for fu, operation, ii, delay in (_canonicalize_executability_edge(graph, edge) for edge in graph.edges)
        if operation in used_operations and operation not in _STRUCTURAL_EXECUTABILITY_OPS
    ]
    retained_fus = list(dict.fromkeys(fu for fu, _, _, _ in retained_edges))
    retained_operations = list(dict.fromkeys(operation for _, operation, _, _ in retained_edges))

    structural_operations = [operation for operation in sorted(used_operations) if operation in _STRUCTURAL_EXECUTABILITY_OPS]
    if structural_operations and _CONTROL_FU not in retained_fus:
        retained_fus.append(_CONTROL_FU)
    retained_operations.extend(operation for operation in structural_operations if operation not in retained_operations)

    region_id = _unique_region_id(taken_region_ids, "executability_graph")
    vertex_ids: dict[tuple[str, str], str] = {}
    nodes: list[UHIRNode] = []
    for fu in retained_fus:
        node_id = _unique_node_id(taken_node_ids, fu, "EXG_FU")
        vertex_ids[("fu", fu)] = node_id
        attributes = {"partition": "fu"}
        if node_id != fu:
            attributes["name"] = fu
        nodes.append(UHIRNode(node_id, "fu", attributes=attributes))
    for operation in retained_operations:
        node_id = _unique_node_id(taken_node_ids, operation, "exg_op")
        vertex_ids[("op", operation)] = node_id
        attributes = {"partition": "op"}
        if node_id != operation:
            attributes["name"] = operation
        nodes.append(UHIRNode(node_id, "op", attributes=attributes))
    edges = [
        UHIREdge("exg", vertex_ids[("fu", fu)], vertex_ids[("op", operation)], {"ii": ii, "d": delay}, directed=False)
        for fu, operation, ii, delay in retained_edges
    ]
    edges.extend(
        UHIREdge("exg", vertex_ids[("fu", _CONTROL_FU)], vertex_ids[("op", operation)], {"ii": 0, "d": 0}, directed=False)
        for operation in structural_operations
    )
    return [UHIRRegion(id=region_id, kind="executability", nodes=nodes, edges=edges)]


def _unique_region_id(taken_ids: set[str], base: str) -> str:
    if base not in taken_ids:
        taken_ids.add(base)
        return base
    suffix = 1
    while f"{base}_{suffix}" in taken_ids:
        suffix += 1
    unique = f"{base}_{suffix}"
    taken_ids.add(unique)
    return unique


def _unique_node_id(taken_ids: set[str], preferred: str, prefix: str) -> str:
    if preferred not in taken_ids:
        taken_ids.add(preferred)
        return preferred
    suffix = 0
    while True:
        candidate = f"{prefix}_{suffix}"
        if candidate not in taken_ids:
            taken_ids.add(candidate)
            return candidate
        suffix += 1


def _used_executability_operations(regions: list[UHIRRegion]) -> set[str]:
    operations: set[str] = set()
    for region in regions:
        if region.kind == "executability":
            continue
        for node in region.nodes:
            operation = _embedded_executability_opcode(node.opcode)
            if operation is not None:
                operations.add(operation)
    return operations


def _canonicalize_executability_edge(
    graph: ExecutabilityGraph,
    edge: tuple[str, str, int, int],
) -> tuple[str, str, int, int]:
    source, target, ii, delay = edge
    if source in graph.functional_units:
        return source, target, ii, delay
    return target, source, ii, delay


def _allocate_region(
    region: UHIRRegion,
    candidates: dict[str, list[AllocationCandidate]],
    value_types: dict[str, str],
    memory_impls: dict[str, object],
    *,
    algorithm: str,
) -> UHIRRegion:
    allocated = UHIRRegion(id=region.id, kind=region.kind, parent=region.parent)
    allocated.region_refs = [UHIRRegionRef(ref.target) for ref in region.region_refs]
    allocated.nodes = [
        _allocate_node(node, candidates, value_types, memory_impls, algorithm=algorithm) for node in region.nodes
    ]
    allocated.edges = [UHIREdge(edge.kind, edge.source, edge.target, dict(edge.attributes)) for edge in region.edges]
    allocated.mappings = [UHIRSourceMap(mapping.node_id, mapping.source_id) for mapping in region.mappings]
    allocated.value_bindings = [
        UHIRValueBinding(binding.producer, binding.register, binding.live_intervals)
        for binding in region.value_bindings
    ]
    allocated.muxes = [UHIRMux(mux.id, mux.inputs, mux.output, mux.select, dict(mux.attributes)) for mux in region.muxes]
    allocated.steps = region.steps
    allocated.latency = region.latency
    allocated.initiation_interval = region.initiation_interval
    return allocated


def _allocate_node(
    node: UHIRNode,
    candidates: dict[str, list[AllocationCandidate]],
    value_types: dict[str, str],
    memory_impls: dict[str, object],
    *,
    algorithm: str,
) -> UHIRNode:
    attributes = dict(node.attributes)
    fixed = _FIXED_ALLOCATIONS.get(node.opcode)
    if fixed is not None:
        attributes["class"], attributes["ii"], attributes["delay"] = fixed
        return UHIRNode(node.id, node.opcode, node.operands, node.result_type, attributes)

    operation = _executability_opcode(node.opcode)
    if operation is None:
        attributes["class"], attributes["ii"], attributes["delay"] = (_CONTROL_FU, 0, 0)
        return UHIRNode(node.id, node.opcode, node.operands, node.result_type, attributes)

    if node.opcode in {"load", "store"} and node.operands:
        memory_choice = memory_impls.get(node.operands[0])
        if memory_choice is not None:
            attributes["class"] = memory_choice.component_spec
            if node.opcode == "load":
                attributes["ii"] = memory_choice.load_ii
                attributes["delay"] = memory_choice.load_delay
            else:
                attributes["ii"] = memory_choice.store_ii
                attributes["delay"] = memory_choice.store_delay
            return UHIRNode(node.id, node.opcode, node.operands, node.result_type, attributes)

    node_candidates = candidates.get(operation)
    if node_candidates is None:
        raise ValueError(f"no executability allocation available for node '{node.id}' with operation '{operation}'")
    compatible = [
        (_infer_candidate_params(candidate, node, value_types), candidate)
        for candidate in node_candidates
    ]
    compatible = [(params, candidate) for params, candidate in compatible if params is not None]
    if not compatible:
        raise ValueError(
            f"no typed executability allocation available for node '{node.id}' with operation '{operation}'"
        )
    chosen_params, chosen_candidate = _choose_typed_allocation_candidate(compatible, algorithm=algorithm)
    attributes["class"] = format_component_spec(chosen_candidate.functional_unit, chosen_params)
    attributes["ii"] = chosen_candidate.ii
    attributes["delay"] = chosen_candidate.delay
    return UHIRNode(node.id, node.opcode, node.operands, node.result_type, attributes)


def _choose_typed_allocation_candidate(
    compatible: list[tuple[dict[str, str], AllocationCandidate]],
    *,
    algorithm: str,
) -> tuple[dict[str, str], AllocationCandidate]:
    chosen_params, chosen_candidate = min(
        compatible,
        key=lambda item: (
            item[1].delay if algorithm == "min_delay" else item[1].ii,
            item[1].ii if algorithm == "min_delay" else item[1].delay,
            item[1].functional_unit,
            tuple(sorted(item[0].items())),
        ),
    )
    return chosen_params, chosen_candidate


def _design_value_types(design: UHIRDesign) -> dict[str, str]:
    value_types: dict[str, str] = {}
    for port in [*design.inputs, *design.outputs]:
        value_types[port.name] = port.type
    for constant in design.constants:
        value_types[constant.name] = constant.type
    for region in design.regions:
        for node in region.nodes:
            if node.result_type is not None:
                value_types[node.id] = node.result_type
    return value_types


def _infer_candidate_params(
    candidate: AllocationCandidate,
    node: UHIRNode,
    value_types: dict[str, str],
) -> dict[str, str] | None:
    if not candidate.type_constraints:
        return {}
    node_types = _node_binding_types(node, value_types)
    inferred: dict[str, str] = {}
    for binding_name, expected_type in candidate.type_constraints:
        actual_type = node_types.get(binding_name)
        if actual_type is None:
            return None
        if _looks_like_concrete_type(expected_type):
            if actual_type != expected_type:
                return None
            continue
        previous = inferred.get(expected_type)
        if previous is None:
            inferred[expected_type] = actual_type
        elif previous != actual_type:
            return None
    return inferred


def _node_binding_types(node: UHIRNode, value_types: dict[str, str]) -> dict[str, str]:
    binding_types: dict[str, str] = {}
    if node.result_type is not None:
        binding_types["result"] = node.result_type
    operand_types: list[str | None] = []
    for operand in node.operands:
        operand_types.append(_operand_type(operand, node, value_types))
    inferred_literal_type = _infer_literal_operand_type(operand_types, node)
    for index, operand_type in enumerate(operand_types):
        if operand_type is None:
            operand_type = inferred_literal_type
        if operand_type is not None:
            binding_types[f"operand{index}"] = operand_type
    return binding_types


def _operand_type(operand: str, node: UHIRNode, value_types: dict[str, str]) -> str | None:
    resolved = value_types.get(operand)
    if resolved is not None:
        return resolved
    typed_literal = _typed_literal_type(operand)
    if typed_literal is not None:
        return typed_literal
    if _looks_like_literal_operand(operand):
        return None
    if node.opcode in {"load", "store"} and operand == node.operands[0]:
        return None
    return value_types.get(operand)


def _infer_literal_operand_type(operand_types: list[str | None], node: UHIRNode) -> str | None:
    resolved = [operand_type for operand_type in operand_types if operand_type is not None]
    if len(set(resolved)) == 1 and resolved:
        return resolved[0]
    if node.result_type is not None and node.opcode in {
        "add",
        "sub",
        "mul",
        "div",
        "mod",
        "and",
        "or",
        "xor",
        "shl",
        "shr",
        "mov",
        "neg",
        "not",
    }:
        return node.result_type
    return None


def _typed_literal_type(operand: str) -> str | None:
    head, sep, tail = operand.partition(":")
    if sep != ":":
        return None
    if not head:
        return None
    return tail or None


def _looks_like_literal_operand(operand: str) -> bool:
    typed = _typed_literal_type(operand)
    if typed is not None:
        return True
    text = operand[1:] if operand.startswith("-") else operand
    return text.isdigit() or operand in {"true", "false"}


def _looks_like_concrete_type(type_name: str) -> bool:
    return bool(type_name) and type_name[0] in {"i", "u"} and type_name[1:].isdigit()


def _design_memory_implementations(
    design: UHIRDesign,
    *,
    memory_policy: MemoryPolicy | None,
    memory_vendor_components: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    policy = MemoryPolicy() if memory_policy is None else memory_policy
    memory_ports = {
        port.name: _parse_memref_type(port.type)
        for port in [*design.inputs, *design.outputs]
        if port.type.startswith("memref<")
    }
    if not memory_ports:
        return {}
    return {
        memory_name: select_memory_implementation(
            component_name="GEN_FF_MEM",
            element_type=element_type,
            depth_words=depth_words,
            policy=policy,
            vendor_components=memory_vendor_components,
        )
        for memory_name, (element_type, depth_words) in memory_ports.items()
    }


def _parse_memref_type(type_name: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"memref<\s*([A-Za-z_][\w$<>]*)\s*(?:,\s*(\d+)\s*)?>", type_name)
    if match is None:
        raise ValueError(f"invalid memref type '{type_name}'")
    element_type, extent_text = match.groups()
    return element_type, None if extent_text is None else int(extent_text)


def _normalize_weight(weight: object) -> tuple[int, int]:
    if isinstance(weight, int):
        return (1, int(weight))
    if isinstance(weight, Mapping):
        ii = weight.get("ii")
        delay = weight.get("d")
        if not isinstance(ii, int) or not isinstance(delay, int):
            raise ValueError(f"executability edge weight must define integer ii/d fields, got {weight!r}")
        return (ii, delay)
    raise ValueError(f"unsupported executability edge weight {weight!r}")


def _executability_opcode(opcode: str) -> str | None:
    if opcode in {"CALL", "call"}:
        return "call"
    if opcode in {"NOP", "nop", "BRANCH", "branch", "LOOP", "loop", "sel"}:
        return None
    return opcode


def _embedded_executability_opcode(opcode: str) -> str | None:
    if opcode in {"NOP", "nop", "BRANCH", "branch", "LOOP", "loop", "CALL", "call", "sel"}:
        return opcode.lower()
    operation = _executability_opcode(opcode)
    return None if operation is None else _normalize_operation_name(operation)


def _normalize_fu_name(name: object) -> str:
    return str(name).upper()


def _normalize_operation_name(name: object) -> str:
    return str(name).lower()


def _clone_port(port: UHIRPort) -> UHIRPort:
    return UHIRPort(port.direction, port.name, port.type)


def _clone_constant(constant: UHIRConstant) -> UHIRConstant:
    return UHIRConstant(constant.name, constant.value, constant.type)


def _clone_schedule(schedule: UHIRSchedule | None) -> UHIRSchedule | None:
    if schedule is None:
        return None
    return UHIRSchedule(schedule.kind)


def _clone_resource(resource: UHIRResource) -> UHIRResource:
    return UHIRResource(resource.kind, resource.id, resource.value, resource.target)
