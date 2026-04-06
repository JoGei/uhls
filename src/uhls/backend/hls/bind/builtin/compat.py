"""Built-in compatibility-based operation binding.

This binder is the dynamic/symbolic counterpart to the concrete interval-based
``left_edge`` binder.

It intentionally prioritizes functional-unit binding and only performs one
conservative dynamic register-binding slice:

* symbolic sched timing is accepted
* concrete interval overlap is used when available
* otherwise hierarchy/control compatibility is used conservatively
* register binding is only added for values in the global storage domain that
  live in mutually exclusive branch arms
* loop/call activation domains remain intentionally conservative

Two operation occurrences may share one FU when all of the following hold:

* they live in different mutually exclusive branch-choice domains, or
* they belong to different loop domains, or
* they are ordered by local same-region precedence, or
* they have concrete non-overlapping occupied intervals

If none of those facts proves compatibility, the binder conservatively treats
the pair as conflicting.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from uhls.backend.uhir.model import UHIRDesign, UHIRNode, UHIRRegion, UHIRResource, UHIRValueBinding
from uhls.backend.uhir.timing import TimingExpr
from uhls.utils.graph import greedy_color_graph, intervals_overlap

from ..interfaces import BindingOccurrence, OperationBinderBase, OperationBindingResult


@dataclass(slots=True, frozen=True)
class CompatibilityOccurrence:
    """One operation occurrence for compatibility-based binding."""

    entity_id: str
    class_name: str
    region_id: str
    node_id: str
    start: int | None
    end: int | None
    domain: str
    branch_choices: frozenset[tuple[str, str]]


@dataclass(slots=True, frozen=True)
class CompatibilityValueInfo:
    """One conservative dynamic register-binding candidate."""

    entity_id: str
    value_id: str
    value_type: str
    region_id: str
    live_start: int
    live_end: int
    storage_domain: str
    branch_choices: frozenset[tuple[str, str]]


@dataclass(slots=True)
class CompatibilityBinder(OperationBinderBase):
    """Bind operations with hierarchy/control compatibility for symbolic schedules."""

    flatten: bool = False

    def bind_operations(self, design: UHIRDesign) -> OperationBindingResult:
        self.validate_sched_stage(design)
        if self.flatten:
            raise ValueError("compat binding does not support --flatten; use left_edge for fully static flattened binding")
        grouped_nodes = self._collect_operation_occurrences(design)
        grouped_values, value_infos = self._collect_value_occurrences(design)
        reachability = self._build_region_reachability(design)

        resources: list[UHIRResource] = []
        node_bindings: dict[str, str] = {}
        value_bindings: list[UHIRValueBinding] = []
        next_resource_index: dict[str, int] = defaultdict(int)
        next_register_index: dict[str, int] = defaultdict(int)

        for class_name, entity_occurrences in sorted(grouped_nodes.items()):
            conflicts = self._build_entity_conflicts(entity_occurrences, reachability)
            colors = greedy_color_graph(
                entity_occurrences,
                conflicts,
                key=lambda entity_id: self._occurrence_order_key(entity_id, entity_occurrences[entity_id]),
            )
            color_to_resource: dict[int, str] = {}
            for color in sorted(set(colors.values())):
                resource_id = f"{class_name.lower()}{next_resource_index[class_name]}"
                next_resource_index[class_name] += 1
                color_to_resource[color] = resource_id
                resources.append(UHIRResource("fu", resource_id, class_name))
            for node_id in entity_occurrences:
                node_bindings[node_id] = color_to_resource[colors[node_id]]

        for value_type, entity_occurrences in sorted(grouped_values.items()):
            if not entity_occurrences:
                continue
            colors = greedy_color_graph(
                entity_occurrences,
                self.build_entity_conflicts(entity_occurrences),
                key=lambda entity_id: self.occurrence_order_key(entity_id, entity_occurrences[entity_id]),
            )
            color_to_register: dict[int, str] = {}
            for color in sorted(set(colors.values())):
                resource_id = _register_resource_id(value_type, next_register_index[value_type])
                next_register_index[value_type] += 1
                color_to_register[color] = resource_id
                resources.append(UHIRResource("reg", resource_id, value_type))
            for entity_id, info in sorted(value_infos.items()):
                if info.value_type != value_type:
                    continue
                register_id = color_to_register[colors[entity_id]]
                value_bindings.append(UHIRValueBinding(info.value_id, register_id, ((info.live_start, info.live_end),)))

        return OperationBindingResult(
            tuple(resources),
            node_bindings,
            tuple(value_bindings),
            metadata={"mode": "compat", "storage_model": "branch_exclusive_only"},
        )

    def _collect_operation_occurrences(
        self,
        design: UHIRDesign,
    ) -> dict[str, dict[str, list[CompatibilityOccurrence]]]:
        grouped: dict[str, dict[str, list[CompatibilityOccurrence]]] = defaultdict(lambda: defaultdict(list))

        def visit_region(region, offset, branch_choices, loop_domain) -> None:
            for node in region.nodes:
                class_name = node.attributes.get("class")
                if not isinstance(class_name, str) or class_name == "CTRL":
                    continue
                start = node.attributes.get("start")
                end = node.attributes.get("end")
                concrete_start = start if isinstance(start, int) else None
                concrete_end = end if isinstance(end, int) else None
                if (concrete_start is None) != (concrete_end is None):
                    raise ValueError(
                        f"bindable node '{region.id}/{node.id}' must use either fully concrete or fully symbolic start/end"
                    )
                grouped[class_name][node.id].append(
                    CompatibilityOccurrence(
                        entity_id=node.id,
                        class_name=class_name,
                        region_id=region.id,
                        node_id=node.id,
                        start=None if concrete_start is None else concrete_start + offset,
                        end=None if concrete_end is None else concrete_end + offset,
                        domain="global" if loop_domain is None else f"loop:{loop_domain}",
                        branch_choices=frozenset(branch_choices),
                    )
                )

        self._walk_design_occurrences(design, flatten=False, visit_region=visit_region)
        return {class_name: dict(entity_occurrences) for class_name, entity_occurrences in grouped.items()}

    def _collect_value_occurrences(
        self,
        design: UHIRDesign,
    ) -> tuple[dict[str, dict[str, list[BindingOccurrence]]], dict[str, CompatibilityValueInfo]]:
        """Collect conservative dynamic register-binding candidates.

        Phase 1 dynamic register binding only reuses storage across mutually
        exclusive branch arms. Any value that lives inside one loop/call
        activation domain is intentionally excluded until overlap domains are
        modeled explicitly.
        """
        grouped: dict[str, dict[str, list[BindingOccurrence]]] = defaultdict(lambda: defaultdict(list))
        value_infos: dict[str, CompatibilityValueInfo] = {}

        def visit_region(region, offset, branch_choices, loop_domain) -> None:
            for node in region.nodes:
                value_type = node.result_type
                class_name = node.attributes.get("class")
                if value_type is None or not isinstance(class_name, str) or class_name == "CTRL":
                    continue
                consumers = self.get_local_value_consumers(region, node)
                if not consumers:
                    continue
                try:
                    live_start, live_end = self.get_value_interval(region, node, consumers)
                except ValueError:
                    continue
                storage_domain = self._value_storage_domain(design, region, loop_domain)
                if storage_domain != "global":
                    continue
                value_id = self.get_value_id(region, node)
                grouped[value_type][node.id].append(
                    BindingOccurrence(
                        entity_id=node.id,
                        class_name=value_type,
                        start=live_start + offset,
                        end=live_end + offset,
                        domain=storage_domain,
                        branch_choices=frozenset(branch_choices),
                    )
                )
                value_infos[node.id] = CompatibilityValueInfo(
                    entity_id=node.id,
                    value_id=value_id,
                    value_type=value_type,
                    region_id=region.id,
                    live_start=live_start,
                    live_end=live_end,
                    storage_domain=storage_domain,
                    branch_choices=frozenset(branch_choices),
                )

        self._walk_design_occurrences(design, flatten=False, visit_region=visit_region)
        return ({value_type: dict(entity_occurrences) for value_type, entity_occurrences in grouped.items()}, value_infos)

    def _build_region_reachability(self, design: UHIRDesign) -> dict[str, dict[str, set[str]]]:
        reachability: dict[str, dict[str, set[str]]] = {}
        for region in design.regions:
            adjacency = {node.id: [] for node in region.nodes}
            for edge in self.iter_region_data_edges(region):
                adjacency[edge.source].append(edge.target)
            region_reachability: dict[str, set[str]] = {}
            for node in region.nodes:
                seen: set[str] = set()
                stack = list(adjacency[node.id])
                while stack:
                    current = stack.pop()
                    if current in seen:
                        continue
                    seen.add(current)
                    stack.extend(adjacency.get(current, ()))
                region_reachability[node.id] = seen
            reachability[region.id] = region_reachability
        return reachability

    def _build_entity_conflicts(
        self,
        entity_occurrences: dict[str, list[CompatibilityOccurrence]],
        reachability: dict[str, dict[str, set[str]]],
    ) -> dict[str, set[str]]:
        conflicts = {entity_id: set() for entity_id in entity_occurrences}
        entity_ids = sorted(entity_occurrences)
        for index, left_id in enumerate(entity_ids):
            left_occurrences = entity_occurrences[left_id]
            for right_id in entity_ids[index + 1 :]:
                if any(
                    self._occurrences_may_conflict(left_occurrence, right_occurrence, reachability)
                    for left_occurrence in left_occurrences
                    for right_occurrence in entity_occurrences[right_id]
                ):
                    conflicts[left_id].add(right_id)
                    conflicts[right_id].add(left_id)
        return conflicts

    def _occurrences_may_conflict(
        self,
        left: CompatibilityOccurrence,
        right: CompatibilityOccurrence,
        reachability: dict[str, dict[str, set[str]]],
    ) -> bool:
        """Return whether two dynamic occurrences may need distinct FUs."""
        if left.domain != right.domain:
            return False
        if not self._branch_choices_compatible(left.branch_choices, right.branch_choices):
            return False
        if left.region_id == right.region_id:
            reachable = reachability.get(left.region_id, {})
            if right.node_id in reachable.get(left.node_id, set()) or left.node_id in reachable.get(right.node_id, set()):
                return False
        if left.start is not None and left.end is not None and right.start is not None and right.end is not None:
            return intervals_overlap((left.start, left.end), (right.start, right.end))
        return True

    def _occurrence_order_key(
        self,
        entity_id: str,
        occurrences: list[CompatibilityOccurrence],
    ) -> tuple[int, int, str]:
        first = min(
            (
                occurrence.start if occurrence.start is not None else 0,
                occurrence.end if occurrence.end is not None else 0,
                occurrence.region_id,
            )
            for occurrence in occurrences
        )
        return first[0], first[1], entity_id

    def _value_storage_domain(
        self,
        design: UHIRDesign,
        region: UHIRRegion,
        loop_domain: str | None,
    ) -> str:
        """Return one conservative dynamic storage domain for one value region."""
        if loop_domain is not None:
            return f"activation:loop:{loop_domain}"
        current = region
        while current.parent is not None:
            parent = design.get_region(current.parent)
            if parent is None:
                break
            owner = _find_child_owner(parent, current.id)
            if owner is not None and owner.opcode in {"loop", "call"}:
                return f"activation:{owner.opcode}:{owner.id}"
            current = parent
        return "global"


def _find_child_owner(parent: UHIRRegion, child_region_id: str) -> UHIRNode | None:
    """Return the hierarchy node that owns one child region, if any."""
    for node in parent.nodes:
        for attr_name in ("child", "true_child", "false_child"):
            if node.attributes.get(attr_name) == child_region_id:
                return node
    return None


def _register_resource_id(value_type: str, index: int) -> str:
    normalized = "".join(char if char.isalnum() else "_" for char in value_type).strip("_").lower()
    if not normalized:
        normalized = "value"
    return f"r_{normalized}_{index}"
