"""Built-in left-edge operation binding."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from uhls.backend.hls.uhir.model import UHIRDesign, UHIRResource, UHIRValueBinding
from uhls.utils.graph import greedy_color_graph

from ..interfaces import BindingOccurrence, OperationBinderBase, OperationBindingResult


@dataclass(slots=True)
class LeftEdgeBinder(OperationBinderBase):
    """Bind scheduled operations conservatively with left-edge interval partitioning."""

    flatten: bool = False

    def bind_operations(self, design: UHIRDesign) -> OperationBindingResult:
        self.validate_sched_design(design)
        if self.flatten:
            self.assert_fully_static_design(design)

        grouped_nodes = self.collect_operation_occurrences(design, flatten=self.flatten)
        grouped_values = self.collect_value_occurrences(design, flatten=self.flatten)
        producer_info = {
            producer.id: (region, producer, self.get_value_id(region, producer))
            for region, producer, _ in self.iter_bindable_values(design)
        }

        resources: list[UHIRResource] = []
        node_bindings: dict[str, str] = {}
        value_bindings: list[UHIRValueBinding] = []
        next_resource_index: dict[str, int] = defaultdict(int)
        next_register_index: dict[str, int] = defaultdict(int)

        for class_name, entity_occurrences in sorted(grouped_nodes.items()):
            colors = greedy_color_graph(
                entity_occurrences,
                self.build_entity_conflicts(entity_occurrences),
                key=lambda entity_id: self.occurrence_order_key(entity_id, entity_occurrences[entity_id]),
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
            if self.flatten:
                occurrence_items = _flattened_value_occurrence_items(entity_occurrences)
                colors = greedy_color_graph(
                    occurrence_items,
                    _flattened_value_conflicts(occurrence_items, self.occurrences_conflict),
                    key=lambda item: self.occurrence_order_key(item[0], [item[1]]),
                    node_key=lambda item: item[0],
                )
            else:
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
            if self.flatten:
                producer_register_intervals: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
                for occurrence_id, occurrence in occurrence_items:
                    producer_id = occurrence.entity_id
                    _, _, value_id = producer_info[producer_id]
                    register_id = color_to_register[colors[occurrence_id]]
                    producer_register_intervals[(value_id, register_id)].append((occurrence.start, occurrence.end))
                for (value_id, register_id), intervals in sorted(producer_register_intervals.items()):
                    value_bindings.append(UHIRValueBinding(value_id, register_id, tuple(sorted(intervals))))
                continue
            for region, producer, consumers in self.iter_bindable_values(design):
                if self.get_value_type(region, producer) != value_type:
                    continue
                register_id = color_to_register[colors[producer.id]]
                live_start, live_end = self.get_value_interval(region, producer, consumers)
                value_bindings.append(
                    UHIRValueBinding(self.get_value_id(region, producer), register_id, ((live_start, live_end),))
                )

        return OperationBindingResult(
            tuple(resources),
            node_bindings,
            tuple(value_bindings),
            metadata={"mode": "flatten" if self.flatten else "hierarchical"},
        )


def _register_resource_id(value_type: str, index: int) -> str:
    normalized = "".join(char if char.isalnum() else "_" for char in value_type).strip("_").lower()
    if not normalized:
        normalized = "value"
    return f"r_{normalized}_{index}"


def _flattened_value_occurrence_items(
    entity_occurrences: dict[str, list[BindingOccurrence]],
) -> list[tuple[str, BindingOccurrence]]:
    items: list[tuple[str, BindingOccurrence]] = []
    for entity_id, occurrences in sorted(entity_occurrences.items()):
        for index, occurrence in enumerate(sorted(occurrences, key=lambda item: (item.start, item.end, item.domain))):
            items.append((f"{entity_id}@{index}", occurrence))
    return items


def _flattened_value_conflicts(
    occurrence_items: list[tuple[str, BindingOccurrence]],
    occurrence_conflicts,
) -> dict[str, set[str]]:
    conflicts = {occurrence_id: set() for occurrence_id, _ in occurrence_items}
    for index, (left_id, left_occurrence) in enumerate(occurrence_items):
        for right_id, right_occurrence in occurrence_items[index + 1 :]:
            if not occurrence_conflicts(left_occurrence, right_occurrence):
                continue
            conflicts[left_id].add(right_id)
            conflicts[right_id].add(left_id)
    return conflicts
