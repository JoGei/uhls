"""Interfaces for pluggable operation binders."""

from __future__ import annotations

from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol

from uhls.backend.hls.uhir.model import UHIREdge, UHIRDesign, UHIRNode, UHIRRegion, UHIRResource, UHIRValueBinding
from uhls.backend.hls.uhir.timing import TimingExpr
from uhls.utils.graph import intervals_overlap

_NON_BINDABLE_CLASSES = frozenset({"CTRL", "ADAPT"})


@dataclass(slots=True, frozen=True)
class OperationBindingResult:
    """One binding result for one scheduled design."""

    resources: tuple[UHIRResource, ...]
    node_bindings: dict[str, str]
    value_bindings: tuple[UHIRValueBinding, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BindingOccurrence:
    """One scheduled occurrence of one bindable operation or value."""

    entity_id: str
    class_name: str
    start: int
    end: int
    domain: str
    branch_choices: frozenset[tuple[str, str]]


@dataclass(slots=True, frozen=True)
class ValueConsumerRef:
    """One consumer of one produced value, optionally shifted in time."""

    region: UHIRRegion
    node: UHIRNode
    start_shift: int = 0


class OperationBinder(Protocol):
    """Interface implemented by one user-supplied binder."""

    def bind_operations(self, design: UHIRDesign) -> OperationBindingResult:
        """Return one operation/register binding for one scheduled design."""


class OperationBinderBase(ABC):
    """Shared validation helpers for binders."""

    def iter_bindable_nodes(self, design: UHIRDesign):
        """Yield bindable operation nodes grouped by region."""
        for region in design.regions:
            for node in region.nodes:
                class_name = node.attributes.get("class")
                if not isinstance(class_name, str) or class_name in _NON_BINDABLE_CLASSES:
                    continue
                yield region, node

    def validate_sched_stage(self, design: UHIRDesign) -> None:
        """Assert one sched-stage design shell before binding it."""
        if design.stage != "sched":
            raise ValueError(f"operation binding expects sched-stage µhIR input, got stage '{design.stage}'")
        if design.schedule is None:
            raise ValueError("operation binding requires schedule kind=...")

    def validate_sched_design(self, design: UHIRDesign) -> None:
        """Assert one sched-stage design shape before binding it."""
        self.validate_sched_stage(design)
        for region, node in self.iter_bindable_nodes(design):
            start = node.attributes.get("start")
            end = node.attributes.get("end")
            if isinstance(start, TimingExpr) or isinstance(end, TimingExpr):
                raise ValueError(
                    f"operation binding currently requires concrete sched timing; "
                    f"bindable node '{region.id}/{node.id}' has symbolic start/end "
                    f"(use the FU-only 'compat' binder for symbolic schedules)"
                )
            self.get_node_interval(region, node)

    def assert_fully_static_design(self, design: UHIRDesign) -> None:
        """Assert that one scheduled design can be flattened into global occurrences."""
        self.validate_sched_design(design)
        for region in design.regions:
            for node in region.nodes:
                if node.opcode != "loop":
                    continue
                trip_count = node.attributes.get("static_trip_count")
                iter_ii = node.attributes.get("iter_initiation_interval")
                child = node.attributes.get("child")
                if not isinstance(child, str) or not child:
                    raise ValueError(f"loop node '{region.id}/{node.id}' is missing child=... for flattened binding")
                if not isinstance(trip_count, int) or trip_count < 0:
                    raise ValueError(
                        f"flattened binding requires a fully static design; loop node '{region.id}/{node.id}' is missing static_trip_count"
                    )
                if not isinstance(iter_ii, int) or iter_ii < 0:
                    raise ValueError(
                        f"flattened binding requires a fully static design; loop node '{region.id}/{node.id}' is missing iter_initiation_interval"
                    )

    def collect_operation_occurrences(
        self,
        design: UHIRDesign,
        *,
        flatten: bool,
    ) -> dict[str, dict[str, list[BindingOccurrence]]]:
        """Group bindable operation occurrences by class and template node id."""
        grouped: dict[str, dict[str, list[BindingOccurrence]]] = defaultdict(lambda: defaultdict(list))

        def visit_region(region: UHIRRegion, offset: int, branch_choices: tuple[tuple[str, str], ...], loop_domain: str | None) -> None:
            for node in region.nodes:
                class_name = node.attributes.get("class")
                if not isinstance(class_name, str) or class_name in _NON_BINDABLE_CLASSES:
                    continue
                start, end = self.get_node_interval(region, node)
                grouped[class_name][node.id].append(
                    BindingOccurrence(
                        entity_id=node.id,
                        class_name=class_name,
                        start=start + offset,
                        end=end + offset,
                        domain="global" if loop_domain is None else f"loop:{loop_domain}",
                        branch_choices=frozenset(branch_choices),
                    )
                )

        self._walk_design_occurrences(design, flatten=flatten, visit_region=visit_region)
        return {class_name: dict(entity_occurrences) for class_name, entity_occurrences in grouped.items()}

    def collect_value_occurrences(
        self,
        design: UHIRDesign,
        *,
        flatten: bool,
    ) -> dict[str, dict[str, list[BindingOccurrence]]]:
        """Group bindable value occurrences by type and template producer id."""
        grouped: dict[str, dict[str, list[BindingOccurrence]]] = defaultdict(lambda: defaultdict(list))

        def visit_region(region: UHIRRegion, offset: int, branch_choices: tuple[tuple[str, str], ...], loop_domain: str | None) -> None:
            for node in region.nodes:
                if node.result_type is None:
                    continue
                class_name = node.attributes.get("class")
                if not isinstance(class_name, str) or class_name in _NON_BINDABLE_CLASSES:
                    continue
                consumers = (
                    self.get_flattened_value_consumers(design, region, node)
                    if flatten
                    else self.get_value_consumers(design, region, node)
                )
                if not consumers:
                    continue
                live_start, live_end = self.get_value_interval(region, node, consumers)
                value_type = self.get_value_type(region, node)
                grouped[value_type][node.id].append(
                    BindingOccurrence(
                        entity_id=node.id,
                        class_name=value_type,
                        start=live_start + offset,
                        end=live_end + offset,
                        domain="global" if loop_domain is None else f"loop:{loop_domain}",
                        branch_choices=frozenset(branch_choices),
                    )
                )

        self._walk_design_occurrences(design, flatten=flatten, visit_region=visit_region)
        return {value_type: dict(entity_occurrences) for value_type, entity_occurrences in grouped.items()}

    def get_flattened_value_consumers(self, design: UHIRDesign, region: UHIRRegion, node: UHIRNode) -> list[ValueConsumerRef]:
        """Return one flattened/static value-consumer set for one producer occurrence.

        Flattened binding should reason about concrete scheduled occurrences, not
        abstract hierarchical merge points. In particular, loop-header values
        should not inherit parent merge-phi consumers for every entered
        iteration, because that would stretch each occurrence all the way to the
        post-loop merge.
        """
        consumers = self.get_local_value_consumers(region, node)
        consumer_names = self.get_value_names(region, node)
        if node.opcode not in {"loop", "branch", "call"}:
            consumers.extend(self._hierarchical_child_consumers(design, region, consumer_names))
        if region.kind != "loop":
            consumers.extend(self._hierarchical_value_consumers(design, region, consumer_names))
        return consumers

    def build_entity_conflicts(
        self,
        entity_occurrences: dict[str, list[BindingOccurrence]],
    ) -> dict[str, set[str]]:
        """Return one symmetric entity conflict graph from scheduled occurrences."""
        conflicts = {entity_id: set() for entity_id in entity_occurrences}

        for entity_id, occurrences in entity_occurrences.items():
            for left, right in combinations(occurrences, 2):
                if not self.occurrences_conflict(left, right):
                    continue
                raise ValueError(
                    f"binding entity '{entity_id}' has overlapping scheduled occurrences and would need replicated instances"
                )

        entity_ids = sorted(entity_occurrences)
        for index, left_id in enumerate(entity_ids):
            left_occurrences = entity_occurrences[left_id]
            for right_id in entity_ids[index + 1 :]:
                if any(
                    self.occurrences_conflict(left_occurrence, right_occurrence)
                    for left_occurrence in left_occurrences
                    for right_occurrence in entity_occurrences[right_id]
                ):
                    conflicts[left_id].add(right_id)
                    conflicts[right_id].add(left_id)
        return conflicts

    def occurrence_order_key(self, entity_id: str, occurrences: list[BindingOccurrence]) -> tuple[int, int, str]:
        """Return one stable left-edge style order key for one binding entity."""
        first = min((occurrence.start, occurrence.end, occurrence.domain) for occurrence in occurrences)
        return first[0], first[1], entity_id

    def occurrences_conflict(self, left: BindingOccurrence, right: BindingOccurrence) -> bool:
        """Return whether two scheduled occurrences may require distinct resources."""
        if left.domain != right.domain:
            return False
        if not self._branch_choices_compatible(left.branch_choices, right.branch_choices):
            return False
        return intervals_overlap((left.start, left.end), (right.start, right.end))

    def get_node_interval(self, region: UHIRRegion, node: UHIRNode) -> tuple[int, int]:
        """Return one bindable node's occupied inclusive interval."""
        start = node.attributes.get("start")
        end = node.attributes.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"bindable node '{region.id}/{node.id}' must declare integer start/end attributes")
        if end < start:
            raise ValueError(f"bindable node '{region.id}/{node.id}' has end < start")
        return start, end

    def get_node_ii(self, region: UHIRRegion, node: UHIRNode) -> int:
        """Return one scheduled node's initiation interval."""
        ii = node.attributes.get("ii")
        if not isinstance(ii, int):
            raise ValueError(f"scheduled node '{region.id}/{node.id}' must declare integer ii")
        if ii < 0:
            raise ValueError(f"scheduled node '{region.id}/{node.id}' has negative ii")
        return ii

    def get_node_class(self, region: UHIRRegion, node: UHIRNode) -> str:
        """Return one bindable node's allocated resource class."""
        class_name = node.attributes.get("class")
        if not isinstance(class_name, str) or not class_name or class_name in _NON_BINDABLE_CLASSES:
            raise ValueError(f"bindable node '{region.id}/{node.id}' must declare one non-CTRL class")
        return class_name

    def _walk_design_occurrences(
        self,
        design: UHIRDesign,
        *,
        flatten: bool,
        visit_region,
    ) -> None:
        region_by_id = {region.id: region for region in design.regions}

        def walk_hierarchical(region_id: str, branch_choices: tuple[tuple[str, str], ...], loop_domain: str | None) -> None:
            region = region_by_id[region_id]
            visit_region(region, 0, branch_choices, loop_domain)
            for node in region.nodes:
                if node.opcode == "loop":
                    child_id = node.attributes.get("child")
                    if isinstance(child_id, str) and child_id:
                        walk_hierarchical(child_id, branch_choices, node.id)
                    continue
                if node.opcode == "branch":
                    true_child = node.attributes.get("true_child")
                    false_child = node.attributes.get("false_child")
                    if isinstance(true_child, str) and true_child:
                        walk_hierarchical(true_child, branch_choices + ((node.id, "true"),), loop_domain)
                    if isinstance(false_child, str) and false_child:
                        walk_hierarchical(false_child, branch_choices + ((node.id, "false"),), loop_domain)
                    continue
                for child_id in _node_children(node):
                    walk_hierarchical(child_id, branch_choices, loop_domain)

        def walk_flattened(region_id: str, offset: int, branch_choices: tuple[tuple[str, str], ...]) -> None:
            region = region_by_id[region_id]
            visit_region(region, offset, branch_choices, None)
            for node in region.nodes:
                if node.opcode == "loop":
                    child_id = node.attributes.get("child")
                    trip_count = node.attributes.get("static_trip_count")
                    iter_ii = node.attributes.get("iter_initiation_interval")
                    node_start = node.attributes.get("start")
                    if (
                        not isinstance(child_id, str)
                        or not isinstance(trip_count, int)
                        or not isinstance(iter_ii, int)
                        or not isinstance(node_start, int)
                    ):
                        raise ValueError(f"loop node '{region.id}/{node.id}' is missing static timing for flattened binding")
                    for iteration in range(trip_count):
                        walk_flattened(child_id, offset + node_start + iteration * iter_ii, branch_choices)
                    continue
                if node.opcode == "branch":
                    true_child = node.attributes.get("true_child")
                    false_child = node.attributes.get("false_child")
                    if region.kind == "loop":
                        if isinstance(true_child, str) and true_child:
                            walk_flattened(true_child, offset, branch_choices + ((node.id, "true"),))
                        # TODO: Expand non-empty loop exit SGUs explicitly once
                        # the loop schedule model materializes them in scroll/flatten mode.
                        continue
                    if isinstance(true_child, str) and true_child:
                        walk_flattened(true_child, offset, branch_choices + ((node.id, "true"),))
                    if isinstance(false_child, str) and false_child:
                        walk_flattened(false_child, offset, branch_choices + ((node.id, "false"),))
                    continue
                for key in ("child", "true_child", "false_child"):
                    child_id = node.attributes.get(key)
                    if not isinstance(child_id, str) or not child_id:
                        continue
                    walk_flattened(child_id, offset + _child_region_shift(node, key), branch_choices)

        roots = sorted(region.id for region in design.regions if region.parent is None)
        if flatten:
            self.assert_fully_static_design(design)
            for region_id in _root_region_ids(design):
                walk_flattened(region_id, 0, ())
            return
        for region_id in roots:
            walk_hierarchical(region_id, (), None)

    def _branch_choices_compatible(
        self,
        left: frozenset[tuple[str, str]],
        right: frozenset[tuple[str, str]],
    ) -> bool:
        left_by_branch = dict(left)
        for branch_id, arm in right:
            other_arm = left_by_branch.get(branch_id)
            if other_arm is not None and other_arm != arm:
                return False
        return True

    def iter_region_data_edges(self, region: UHIRRegion):
        """Yield one region's local data-like edges."""
        node_ids = {node.id for node in region.nodes}
        for edge in region.edges:
            if edge.kind == "seq":
                continue
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(
                    f"region '{region.id}' contains non-local data edge '{edge.source} -> {edge.target}' of kind '{edge.kind}'"
                )
            yield edge

    def get_value_consumers(self, design: UHIRDesign, region: UHIRRegion, node: UHIRNode) -> list[ValueConsumerRef]:
        """Return consumer nodes for one produced value, including hierarchical phi uses."""
        consumers = self.get_local_value_consumers(region, node)
        consumer_names = self.get_value_names(region, node)
        if node.opcode not in {"loop", "branch", "call"}:
            child_consumers = self._hierarchical_child_consumers(design, region, consumer_names)
            consumers.extend(child_consumers)
        parent_consumer = self._hierarchical_value_consumers(design, region, consumer_names)
        consumers.extend(parent_consumer)
        return consumers

    def get_local_value_consumers(self, region: UHIRRegion, node: UHIRNode) -> list[ValueConsumerRef]:
        """Return one produced value's consumers that live inside the same SGU."""
        node_by_id = {candidate.id: candidate for candidate in region.nodes}
        consumers: list[ValueConsumerRef] = []
        for edge in self.iter_region_data_edges(region):
            if edge.source != node.id:
                continue
            consumer = node_by_id[edge.target]
            if consumer.opcode == "nop" and consumer.attributes.get("role") == "sink":
                continue
            consumers.append(ValueConsumerRef(region, consumer))
        return consumers

    def iter_bindable_values(self, design: UHIRDesign):
        """Yield bindable produced values with non-empty live ranges."""
        for region in design.regions:
            for node in region.nodes:
                if node.result_type is None:
                    continue
                class_name = node.attributes.get("class")
                if not isinstance(class_name, str) or class_name in _NON_BINDABLE_CLASSES:
                    continue
                consumers = self.get_value_consumers(design, region, node)
                if not consumers:
                    continue
                yield region, node, consumers

    def get_value_interval(
        self,
        region: UHIRRegion,
        producer: UHIRNode,
        consumers: list[ValueConsumerRef],
    ) -> tuple[int, int]:
        """Return one produced value's live interval."""
        # TODO: If bind later needs CFG-sensitive or phi-aware register
        # liveness, move the reusable fixed-point/dataflow machinery into one
        # shared multi-level utility library and keep this scheduled-interval
        # extraction as the sched/uhIR-specific adapter.
        # TODO: If control-flow reconstruction becomes shared across middleend
        # and backend IRs, add one common control-flow helper layer too.
        _, producer_end = self.get_node_interval(region, producer)
        live_start = producer_end + 1
        live_end = max(
            self.get_node_interval(consumer.region, consumer.node)[0]
            + consumer.start_shift
            + max(self.get_node_ii(consumer.region, consumer.node), 1)
            - 1
            for consumer in consumers
        )
        if live_end < live_start:
            raise ValueError(f"value '{region.id}/{producer.id}' has live_end < live_start")
        return live_start, live_end

    def get_value_type(self, region: UHIRRegion, producer: UHIRNode) -> str:
        """Return one produced value's register type."""
        if producer.result_type is None:
            raise ValueError(f"value '{region.id}/{producer.id}' is missing its result type")
        return producer.result_type

    def get_value_id(self, region: UHIRRegion, producer: UHIRNode) -> str:
        """Return one textual value id for one produced result."""
        for mapping in region.mappings:
            if mapping.node_id == producer.id:
                return mapping.source_id
        return producer.id

    def get_value_binding_id(self, design: UHIRDesign, region: UHIRRegion, producer: UHIRNode) -> str:
        """Return one bind-stage value-binding id, disambiguating colliding mapped names."""
        value_id = self.get_value_id(region, producer)
        if _value_id_is_ambiguous(design, value_id):
            return producer.id
        return value_id

    def get_value_names(self, region: UHIRRegion, producer: UHIRNode) -> set[str]:
        """Return all names that may refer to one produced value."""
        names = {producer.id}
        for mapping in region.mappings:
            if mapping.node_id == producer.id:
                names.add(mapping.source_id)
        return names

    def _hierarchical_value_consumers(
        self,
        design: UHIRDesign,
        region: UHIRRegion,
        consumer_names: set[str],
    ) -> list[ValueConsumerRef]:
        if region.parent is None:
            return []
        parent_region = design.get_region(region.parent)
        if parent_region is None:
            return []
        consumers: list[ValueConsumerRef] = []
        shift = self._parent_consumer_shift(design, region, parent_region)
        for parent_node in parent_region.nodes:
            if parent_node.opcode != "phi":
                continue
            if not any(operand in consumer_names for operand in parent_node.operands):
                continue
            consumers.append(ValueConsumerRef(parent_region, parent_node, shift))
        return consumers

    def _hierarchical_child_consumers(
        self,
        design: UHIRDesign,
        region: UHIRRegion,
        consumer_names: set[str],
    ) -> list[ValueConsumerRef]:
        region_by_id = {candidate.id: candidate for candidate in design.regions}
        consumers: list[ValueConsumerRef] = []
        visited: set[str] = set()

        def visit_child_region(child_region_id: str, accumulated_shift: int) -> None:
            if child_region_id in visited:
                return
            visited.add(child_region_id)
            child_region = region_by_id.get(child_region_id)
            if child_region is None:
                return
            for child_node in child_region.nodes:
                if child_node.opcode == "nop" and child_node.attributes.get("role") in {"source", "sink"}:
                    continue
                if any(operand in consumer_names for operand in child_node.operands):
                    consumers.append(ValueConsumerRef(child_region, child_node, accumulated_shift))
                for key in ("child", "true_child", "false_child"):
                    nested_child_id = child_node.attributes.get(key)
                    if not isinstance(nested_child_id, str) or not nested_child_id:
                        continue
                    visit_child_region(nested_child_id, accumulated_shift + _child_region_shift(child_node, key))

        for node in region.nodes:
            for key in ("child", "true_child", "false_child"):
                child_region_id = node.attributes.get(key)
                if not isinstance(child_region_id, str) or not child_region_id:
                    continue
                visit_child_region(child_region_id, _child_region_shift(node, key))
        return consumers

    def _parent_consumer_shift(self, design: UHIRDesign, region: UHIRRegion, parent_region: UHIRRegion) -> int:
        if parent_region.kind != "loop" or region.kind != "body":
            return 0
        if parent_region.parent is None:
            return 0
        grandparent_region = design.get_region(parent_region.parent)
        if grandparent_region is None:
            return 0
        loop_node = next(
            (
                node
                for node in grandparent_region.nodes
                if node.opcode == "loop" and node.attributes.get("child") == parent_region.id
            ),
            None,
        )
        if loop_node is None:
            return 0
        iter_ii = loop_node.attributes.get("iter_initiation_interval")
        if not isinstance(iter_ii, int):
            raise ValueError(
                f"loop node '{grandparent_region.id}/{loop_node.id}' is missing iter_initiation_interval for loop-carried binding"
            )
        return iter_ii


def _node_children(node: UHIRNode) -> list[str]:
    children: list[str] = []
    for key in ("child", "true_child", "false_child"):
        value = node.attributes.get(key)
        if isinstance(value, str) and value:
            children.append(value)
    return children


def _root_region_ids(design: UHIRDesign) -> list[str]:
    referenced = {
        child_id
        for region in design.regions
        for node in region.nodes
        for child_id in _node_children(node)
    }
    return sorted(
        region.id
        for region in design.regions
        if region.parent is None and region.id not in referenced
    )


def _child_region_shift(node: UHIRNode, key: str) -> int:
    if key != "child":
        return 0
    if node.attributes.get("child_timebase") == "global":
        return 0
    node_start = node.attributes.get("start")
    if isinstance(node_start, int):
        return node_start
    return 0


def _value_id_is_ambiguous(design: UHIRDesign, value_id: str) -> bool:
    if not isinstance(value_id, str) or not value_id:
        return False
    matches = 0
    for region in design.regions:
        local_nodes = {node.id for node in region.nodes}
        if value_id in local_nodes:
            matches += 1
        for mapping in region.mappings:
            if mapping.source_id == value_id and mapping.node_id in local_nodes:
                matches += 1
        if matches > 1:
            return True
    return False
