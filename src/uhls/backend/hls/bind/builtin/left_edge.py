"""Built-in left-edge operation binding."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from uhls.backend.uhir.model import UHIRDesign, UHIRResource, UHIRValueBinding
from uhls.utils.graph import greedy_color_graph

from ..interfaces import OperationBinderBase, OperationBindingResult


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
