"""Analysis dump helpers for bind-stage operation binding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from uhls.backend.uhir.model import UHIRDesign, UHIRNode, UHIRRegion, UHIRValueBinding
from uhls.middleend.uir import COMPACT_OPCODE_LABELS
from uhls.utils.graph import interval_conflicts

from .interfaces import OperationBinderBase, ValueConsumerRef

BIND_DUMP_KINDS = ("compatibility", "conflict", "trp", "trp_unroll", "dfgsb", "dfgsb_unroll")
_BIND_DUMP_ALIASES = {
    "compa": "compatibility",
}


@dataclass(slots=True, frozen=True)
class _BoundOccurrence:
    category: str
    scope_id: str
    region_id: str
    node_id: str
    display_id: str
    opcode: str
    bind: str
    class_name: str
    start: int
    end: int


@dataclass(slots=True)
class _DFGSBTableLayout:
    table_node_id: str
    op_in_ports: dict[str, str]
    op_out_ports: dict[str, str]
    reg_ports: dict[tuple[str, int], str]


@dataclass(slots=True)
class _DFGSBDotLayout:
    cluster_id: str
    op_nodes: dict[tuple[str, str], str]
    reg_nodes: dict[tuple[str, str, int], str]
    source_nodes: dict[str, str]
    output_nodes: dict[str, str]


@dataclass(slots=True, frozen=True)
class _DFGSBSourceSpec:
    key: str
    label: str
    kind: str


@dataclass(slots=True, frozen=True)
class _DFGSBOutputSpec:
    key: str
    label: str
    region_id: str
    node_id: str


def parse_bind_dump_spec(spec_text: str) -> tuple[str, ...]:
    """Parse one comma-separated bind dump spec."""
    requested = tuple(
        _BIND_DUMP_ALIASES.get(item.strip().lower(), item.strip().lower())
        for item in spec_text.split(",")
        if item.strip()
    )
    if not requested:
        raise ValueError("bind dump spec must not be empty")
    unknown = sorted(item for item in requested if item not in BIND_DUMP_KINDS)
    if unknown:
        expected = ", ".join(BIND_DUMP_KINDS)
        raise ValueError(f"unsupported bind dump kind(s): {', '.join(unknown)}; expected one of: {expected}")
    unique: list[str] = []
    seen: set[str] = set()
    for item in requested:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return tuple(unique)


def format_bind_dump(design: UHIRDesign, dump_kinds: tuple[str, ...], *, compact: bool = False) -> str:
    """Pretty-print one or more bind analysis dumps."""
    _validate_bind_design(design)
    sections: list[str] = []
    for index, dump_kind in enumerate(dump_kinds):
        if index:
            sections.append("")
        sections.extend(_format_one_dump(design, dump_kind, compact=compact))
    return "\n".join(sections)


def bind_dump_to_dot(design: UHIRDesign, dump_kinds: tuple[str, ...], *, compact: bool = False) -> str:
    """Render one or more bind analysis dumps as one DOT graph."""
    _validate_bind_design(design)
    lines = [f'digraph "{design.name}.bind.dump" {{', "  compound=true;", "  node [style=filled];"]
    for dump_kind in dump_kinds:
        if dump_kind == "conflict":
            lines.extend(_dot_conflict_like(design, compatibility=False, compact=compact))
        elif dump_kind == "compatibility":
            lines.extend(_dot_conflict_like(design, compatibility=True, compact=compact))
        elif dump_kind == "trp":
            lines.extend(_dot_trp(design, compact=compact))
        elif dump_kind == "trp_unroll":
            lines.extend(_dot_trp_unroll(design, compact=compact))
        elif dump_kind == "dfgsb":
            lines.extend(_dot_dfgsb(design, compact=compact))
        elif dump_kind == "dfgsb_unroll":
            lines.extend(_dot_dfgsb_unroll(design, compact=compact))
        else:
            raise AssertionError(f"unsupported dump kind {dump_kind!r}")
    lines.append("}")
    return "\n".join(lines)


def _format_one_dump(design: UHIRDesign, dump_kind: str, *, compact: bool) -> list[str]:
    if dump_kind == "conflict":
        return _format_conflict_like(design, compatibility=False)
    if dump_kind == "compatibility":
        return _format_conflict_like(design, compatibility=True)
    if dump_kind == "trp":
        return _format_trp(design, compact=compact)
    if dump_kind == "trp_unroll":
        return _format_trp_unroll(design, compact=compact)
    if dump_kind == "dfgsb":
        return _format_dfgsb(design, compact=compact)
    if dump_kind == "dfgsb_unroll":
        return _format_dfgsb_unroll(design, compact=compact)
    raise AssertionError(f"unsupported dump kind {dump_kind!r}")


def _format_conflict_like(design: UHIRDesign, *, compatibility: bool) -> list[str]:
    graph_name = "compatibility" if compatibility else "conflict"
    graph_type = "compatibility graph" if compatibility else "conflict graph"
    lines = [f"bind_dump {graph_name}"]
    for (category, class_name), region_entries in _group_entries(_collect_local_bound_entries(design)).items():
        lines.append(f"class {class_name} ({category} {graph_type})")
        for region_id, entries in sorted(region_entries.items()):
            lines.append(f"  region {region_id}")
            for entry in entries:
                lines.append(
                    f"    node {entry.node_id} opcode={entry.opcode} bind={entry.bind} interval={entry.start}..{entry.end}"
                )
            edges = _compatibility_edges(entries) if compatibility else _conflict_edges(entries)
            if not edges:
                lines.append("    edge <none>")
                continue
            for source_id, target_id in edges:
                lines.append(f"    edge {source_id} -- {target_id}")
    return lines


def _format_trp(design: UHIRDesign, *, compact: bool) -> list[str]:
    lines = ["bind_dump trp"]
    for scope_id, entries in sorted(_group_by_scope(_collect_local_trp_entries(design)).items()):
        lines.append(f"region {scope_id} (time-resource plane)")
        lines.extend(f"  {line}" for line in _render_trp_text(entries, compact=compact))
    return lines


def _format_trp_unroll(design: UHIRDesign, *, compact: bool) -> list[str]:
    entries = _collect_scroll_bound_entries(design)
    lines = ["bind_dump trp_unroll", "global (time-resource plane unrolled)"]
    lines.extend(f"  {line}" for line in _render_trp_text(entries, compact=compact))
    return lines


def _format_dfgsb(design: UHIRDesign, *, compact: bool) -> list[str]:
    region_by_id = {region.id: region for region in design.regions}
    grouped = _group_by_scope(_filter_dfgsb_entries(design, _collect_local_dfgsb_entries(design)))
    lines = ["bind_dump dfgsb"]
    first = True
    for region_id in _hierarchy_region_order(design):
        if not first:
            lines.append("")
        first = False
        region = region_by_id[region_id]
        lines.append(_region_boundary_line(region, region_by_id))
        lines.extend(f"  {line}" for line in _render_dfgsb_text(grouped.get(region_id, []), compact=compact))
    return lines


def _format_dfgsb_unroll(design: UHIRDesign, *, compact: bool) -> list[str]:
    entries = _filter_dfgsb_entries(design, _collect_scroll_bound_entries(design))
    lines = ["bind_dump dfgsb_unroll", "global (dataflow graph with schedule and binding, unrolled)"]
    lines.extend(f"  {line}" for line in _render_dfgsb_text(entries, compact=compact))
    return lines


def _dot_conflict_like(design: UHIRDesign, *, compatibility: bool, compact: bool) -> list[str]:
    graph_name = "compatibility" if compatibility else "conflict"
    graph_type = "compatibility graph" if compatibility else "conflict graph"
    edge_color = "#2ca02c" if compatibility else "#666666"
    lines: list[str] = []
    for (category, class_name), region_entries in _group_entries(_collect_local_bound_entries(design)).items():
        cluster_id = f"{graph_name}_{category}_{class_name.lower()}"
        lines.append(f'  subgraph "cluster_{cluster_id}" {{')
        lines.append(f'    label="{_escape_dot(class_name)} {category} {graph_type}";')
        lines.append("    color=gray70;")
        resource_colors = _resource_color_map([entry.bind for region in region_entries.values() for entry in region])
        for entries in region_entries.values():
            for entry in entries:
                lines.append(
                    f'    "{entry.display_id}" [label="{_escape_dot(_entry_label(entry, compact=compact))}", '
                    f'fillcolor="{resource_colors[entry.bind]}", shape=ellipse];'
                )
        rendered_edges: set[tuple[str, str]] = set()
        for region_id, entries in sorted(region_entries.items()):
            edges = _compatibility_edges(entries) if compatibility else _conflict_edges(entries)
            for source_id, target_id in edges:
                edge_key = tuple(sorted((source_id, target_id)))
                if edge_key in rendered_edges:
                    continue
                rendered_edges.add(edge_key)
                lines.append(
                    f'    "{edge_key[0]}" -> "{edge_key[1]}" [label="{_escape_dot(region_id)}", color="{edge_color}", dir=none];'
                )
        lines.append("  }")
    return lines


def _dot_trp(design: UHIRDesign, *, compact: bool) -> list[str]:
    lines: list[str] = []
    for scope_id, entries in sorted(_group_by_scope(_collect_local_trp_entries(design)).items()):
        cluster_id = f"trp_{scope_id}"
        lines.append(f'  subgraph "cluster_{cluster_id}" {{')
        lines.append(f'    label="{_escape_dot(scope_id)} time-resource plane";')
        lines.append("    color=gray70;")
        lines.append(f'    "table_{cluster_id}" [shape=plain, label=<{_render_trp_html_table(entries, compact=compact)}>];')
        lines.append("  }")
    return lines


def _dot_trp_unroll(design: UHIRDesign, *, compact: bool) -> list[str]:
    entries = _collect_scroll_bound_entries(design)
    cluster_id = "trp_unroll"
    return [
        f'  subgraph "cluster_{cluster_id}" {{',
        '    label="global time-resource plane unrolled";',
        "    color=gray70;",
        f'    "table_{cluster_id}" [shape=plain, label=<{_render_trp_html_table(entries, compact=compact)}>];',
        "  }",
    ]


def _dot_dfgsb(design: UHIRDesign, *, compact: bool) -> list[str]:
    region_by_id = {region.id: region for region in design.regions}
    lines: list[str] = []
    layouts: dict[str, _DFGSBDotLayout] = {}
    grouped = _group_by_scope(_filter_dfgsb_entries(design, _collect_local_dfgsb_entries(design)))
    for region_id in _hierarchy_region_order(design):
        region = region_by_id[region_id]
        cluster_id = f"dfgsb_{region_id}"
        layout_lines, layout = _render_dfgsb_dot_scope(
            cluster_id=cluster_id,
            label=_region_boundary_label(region, region_by_id),
            entries=grouped.get(region_id, []),
            source_specs=_collect_dfgsb_source_specs_from_nodes(design, list(region.nodes)),
            output_specs=_collect_dfgsb_output_specs(design) if region.parent is None else [],
            compact=compact,
        )
        layouts[region_id] = layout
        lines.extend(layout_lines)
    lines.extend(_dot_dfgsb_region_edges(design, layouts, grouped))
    return lines


def _dot_dfgsb_unroll(design: UHIRDesign, *, compact: bool) -> list[str]:
    entries = _filter_dfgsb_entries(design, _collect_scroll_bound_entries(design))
    lines, layout = _render_dfgsb_dot_scope(
        cluster_id="dfgsb_unroll",
        label="global dataflow graph with schedule and binding (unrolled)",
        entries=entries,
        source_specs=[],
        output_specs=_collect_dfgsb_output_specs(design),
        compact=compact,
    )
    lines.extend(_dot_dfgsb_unroll_edges(design, layout, entries))
    return lines


def _collect_local_bound_entries(design: UHIRDesign) -> list[_BoundOccurrence]:
    helper = OperationBinderBase()
    entries: list[_BoundOccurrence] = []
    for region in design.regions:
        local_value_ids = {mapping.source_id for mapping in region.mappings}
        local_value_ids.update(node.id for node in region.nodes)
        for node in region.nodes:
            class_name = node.attributes.get("class")
            if not isinstance(class_name, str):
                continue
            bind = _analysis_bind(node, class_name)
            start, end = helper.get_node_interval(region, node)
            entries.append(
                _BoundOccurrence(
                    category="operation",
                    scope_id=region.id,
                    region_id=region.id,
                    node_id=node.id,
                    display_id=node.id,
                    opcode=node.opcode,
                    bind=bind,
                    class_name=class_name,
                    start=start,
                    end=end,
                )
            )
        for binding in region.value_bindings:
            register_id = binding.register
            register_resource = next((resource for resource in design.resources if resource.id == register_id), None)
            if register_resource is None or register_resource.kind != "reg":
                continue
            if binding.producer not in local_value_ids:
                continue
            for index, (start, end) in enumerate(binding.live_intervals):
                display_id = f"reg_{binding.producer}" if len(binding.live_intervals) == 1 else f"reg_{binding.producer}_{index}"
                entries.append(
                    _BoundOccurrence(
                        category="register",
                        scope_id=region.id,
                        region_id=region.id,
                        node_id=binding.producer,
                        display_id=display_id,
                        opcode="reg",
                        bind=register_id,
                        class_name=register_resource.value,
                        start=start,
                        end=end,
                    )
                )
    return entries


def _collect_local_trp_entries(design: UHIRDesign) -> list[_BoundOccurrence]:
    helper = OperationBinderBase()
    resources_by_id = {resource.id: resource for resource in design.resources}
    entries: list[_BoundOccurrence] = []
    for region in design.regions:
        node_intervals = [
            helper.get_node_interval(region, node)
            for node in region.nodes
            if isinstance(node.attributes.get("start"), int) and isinstance(node.attributes.get("end"), int)
        ]
        if node_intervals:
            region_start = min(start for start, _ in node_intervals)
            region_end = max(end for _, end in node_intervals)
        else:
            region_start = 0
            region_end = 0
        local_value_ids = {mapping.source_id for mapping in region.mappings}
        local_value_ids.update(node.id for node in region.nodes)
        for node in region.nodes:
            class_name = node.attributes.get("class")
            if not isinstance(class_name, str):
                continue
            bind = _analysis_bind(node, class_name)
            start, end = helper.get_node_interval(region, node)
            entries.append(
                _BoundOccurrence(
                    category="operation",
                    scope_id=region.id,
                    region_id=region.id,
                    node_id=node.id,
                    display_id=node.id,
                    opcode=node.opcode,
                    bind=bind,
                    class_name=class_name,
                    start=start,
                    end=end,
                )
            )
        for binding in region.value_bindings:
            register_resource = resources_by_id.get(binding.register)
            if register_resource is None or register_resource.kind != "reg":
                continue
            if binding.producer not in local_value_ids:
                continue
            for index, (start, end) in enumerate(binding.live_intervals):
                seg_start = max(start, region_start)
                seg_end = min(end, region_end)
                if seg_end < seg_start:
                    continue
                display_id = f"reg_{binding.producer}" if len(binding.live_intervals) == 1 else f"reg_{binding.producer}_{index}"
                entries.append(
                    _BoundOccurrence(
                        category="register",
                        scope_id=region.id,
                        region_id=region.id,
                        node_id=binding.producer,
                        display_id=display_id,
                        opcode="reg",
                        bind=binding.register,
                        class_name=register_resource.value,
                        start=seg_start,
                        end=seg_end,
                    )
                )
    return entries


def _collect_local_dfgsb_entries(design: UHIRDesign) -> list[_BoundOccurrence]:
    helper = OperationBinderBase()
    entries: list[_BoundOccurrence] = []
    resources_by_id = {resource.id: resource for resource in design.resources}
    for region in design.regions:
        bindings_by_name: dict[str, list[UHIRValueBinding]] = {}
        for binding in region.value_bindings:
            bindings_by_name.setdefault(binding.producer, []).append(binding)
        region_exit = max(
            (helper.get_node_interval(region, node)[1] for node in region.nodes),
            default=0,
        )
        for node in region.nodes:
            class_name = node.attributes.get("class")
            if not isinstance(class_name, str):
                continue
            bind = _analysis_bind(node, class_name)
            start, end = helper.get_node_interval(region, node)
            entries.append(
                _BoundOccurrence(
                    category="operation",
                    scope_id=region.id,
                    region_id=region.id,
                    node_id=node.id,
                    display_id=node.id,
                    opcode=node.opcode,
                    bind=bind,
                    class_name=class_name,
                    start=start,
                    end=end,
                )
            )

        for node in region.nodes:
            if node.result_type is None:
                continue
            class_name = node.attributes.get("class")
            if not isinstance(class_name, str) or class_name == "CTRL":
                continue
            all_consumers = helper.get_value_consumers(design, region, node)
            if not all_consumers:
                continue
            local_consumers = helper.get_local_value_consumers(region, node)
            producer_end = helper.get_node_interval(region, node)[1]
            live_start = producer_end + 1
            if local_consumers:
                _, live_end = helper.get_value_interval(region, node, local_consumers)
            else:
                live_end = region_exit
            if live_end < live_start:
                continue
            binding_index = 0
            for binding in _matching_value_bindings(region, bindings_by_name, helper.get_value_names(region, node)):
                register_resource = resources_by_id.get(binding.register)
                if register_resource is None or register_resource.kind != "reg":
                    continue
                for interval_index, (start, end) in enumerate(binding.live_intervals):
                    seg_start = max(start, live_start)
                    seg_end = min(end, live_end)
                    if seg_end < seg_start:
                        continue
                    display_id = f"reg_{binding.producer}" if len(binding.live_intervals) == 1 else f"reg_{binding.producer}_{interval_index}"
                    if binding_index:
                        display_id = f"{display_id}_{binding_index}"
                    entries.append(
                        _BoundOccurrence(
                            category="register",
                            scope_id=region.id,
                            region_id=region.id,
                            node_id=binding.producer,
                            display_id=display_id,
                            opcode="reg",
                            bind=binding.register,
                            class_name=register_resource.value,
                            start=seg_start,
                            end=seg_end,
                        )
                    )
                binding_index += 1
    return entries


def _collect_scroll_bound_entries(design: UHIRDesign) -> list[_BoundOccurrence]:
    helper = OperationBinderBase()
    region_by_id = {region.id: region for region in design.regions}
    entries: list[_BoundOccurrence] = []
    flattened_bind = _bind_design_looks_flattened(design, helper)

    if flattened_bind:
        entries.extend(_collect_flattened_register_entries(design))

    def append_entry(entry: _BoundOccurrence, *, max_end: int | None) -> None:
        if max_end is not None and entry.start > max_end:
            return
        if max_end is not None and entry.end > max_end:
            entry = _BoundOccurrence(
                category=entry.category,
                scope_id=entry.scope_id,
                region_id=entry.region_id,
                node_id=entry.node_id,
                display_id=entry.display_id,
                opcode=entry.opcode,
                bind=entry.bind,
                class_name=entry.class_name,
                start=entry.start,
                end=max_end,
            )
        if entry.end < entry.start:
            return
        entries.append(entry)

    def visit_region(region_id: str, offset: int, *, suffix: str = "", max_end: int | None = None) -> None:
        region = region_by_id[region_id]
        local_value_ids = {mapping.source_id for mapping in region.mappings}
        local_value_ids.update(node.id for node in region.nodes)
        for node in region.nodes:
            class_name = node.attributes.get("class")
            if not isinstance(class_name, str):
                continue
            bind = _analysis_bind(node, class_name)
            start, end = helper.get_node_interval(region, node)
            display_id = node.id if not suffix else f"{node.id}{suffix}"
            append_entry(
                _BoundOccurrence(
                    category="operation",
                    scope_id="global",
                    region_id=region.id,
                    node_id=node.id,
                    display_id=display_id,
                    opcode=node.opcode,
                    bind=bind,
                    class_name=class_name,
                    start=start + offset,
                    end=end + offset,
                )
                ,
                max_end=max_end,
            )
        for node in region.nodes:
            node_start = node.attributes.get("start")
            if not isinstance(node_start, int):
                continue
            if node.opcode == "loop":
                child_id = node.attributes.get("child")
                trip_count = node.attributes.get("static_trip_count")
                iter_ii = node.attributes.get("iter_initiation_interval")
                node_end = node.attributes.get("end")
                if not isinstance(child_id, str) or not isinstance(trip_count, int) or not isinstance(iter_ii, int):
                    raise ValueError(
                        f"loop node '{node.id}' requires child/static_trip_count/iter_initiation_interval for unrolled bind dumps"
                    )
                if not isinstance(node_end, int):
                    raise ValueError(f"loop node '{node.id}' requires end for unrolled bind dumps")
                child_max_end = offset + node_end
                for iteration in range(trip_count):
                    visit_loop_header(
                        child_id,
                        offset + node_start + iteration * iter_ii,
                        iteration,
                        expand_body=True,
                        max_end=child_max_end,
                    )
                visit_loop_header(
                    child_id,
                    offset + node_start + trip_count * iter_ii,
                    trip_count,
                    expand_body=False,
                    max_end=child_max_end,
                )
                continue

            for child_id in _node_children(node):
                # Child regions in sched/bind µhIR are already shifted into their
                # parent-local time base, so the scroll expansion only reapplies
                # the enclosing scope offset once.
                visit_region(child_id, offset, suffix=suffix, max_end=max_end)

    def visit_loop_header(
        region_id: str,
        offset: int,
        iteration: int,
        *,
        expand_body: bool,
        max_end: int | None,
    ) -> None:
        region = region_by_id[region_id]
        suffix = f"@{iteration}"
        local_value_ids = {mapping.source_id for mapping in region.mappings}
        local_value_ids.update(node.id for node in region.nodes)
        for node in region.nodes:
            class_name = node.attributes.get("class")
            if not isinstance(class_name, str):
                continue
            bind = _analysis_bind(node, class_name)
            start, end = helper.get_node_interval(region, node)
            append_entry(
                _BoundOccurrence(
                    category="operation",
                    scope_id="global",
                    region_id=region.id,
                    node_id=node.id,
                    display_id=f"{node.id}{suffix}",
                    opcode=node.opcode,
                    bind=bind,
                    class_name=class_name,
                    start=start + offset,
                    end=end + offset,
                ),
                max_end=max_end,
            )
        for node in region.nodes:
            node_start = node.attributes.get("start")
            if not isinstance(node_start, int):
                continue
            if node.opcode == "branch":
                true_child = node.attributes.get("true_child")
                if expand_body and isinstance(true_child, str):
                    visit_region(true_child, offset, suffix=suffix, max_end=max_end)
                # TODO: Add explicit expansion for loop exit regions once they may contain bindable work.
                continue
            for child_id in _node_children(node):
                visit_region(child_id, offset, suffix=suffix, max_end=max_end)

    for region_id in sorted(region.id for region in design.regions if region.parent is None):
        visit_region(region_id, 0)
    if not flattened_bind:
        entries.extend(_collect_hierarchical_unrolled_register_entries(design, entries))
    return entries


def _group_entries(
    entries: list[_BoundOccurrence],
) -> dict[tuple[str, str], dict[str, list[_BoundOccurrence]]]:
    grouped: dict[tuple[str, str], dict[str, list[_BoundOccurrence]]] = {}
    for entry in entries:
        grouped.setdefault((entry.category, entry.class_name), {}).setdefault(entry.region_id, []).append(entry)
    for class_regions in grouped.values():
        for region_entries in class_regions.values():
            region_entries.sort(key=lambda entry: (entry.start, entry.end, entry.display_id))
    return grouped


def _group_by_scope(entries: list[_BoundOccurrence]) -> dict[str, list[_BoundOccurrence]]:
    grouped: dict[str, list[_BoundOccurrence]] = {}
    for entry in entries:
        grouped.setdefault(entry.scope_id, []).append(entry)
    for scope_entries in grouped.values():
        scope_entries.sort(key=lambda entry: (entry.start, entry.end, entry.display_id))
    return grouped


def _filter_dfgsb_entries(design: UHIRDesign, entries: list[_BoundOccurrence]) -> list[_BoundOccurrence]:
    helper = OperationBinderBase()
    region_by_id = {region.id: region for region in design.regions}
    producer_by_region = {region.id: _producer_node_by_name(region) for region in design.regions}
    filtered: list[_BoundOccurrence] = []
    for entry in entries:
        if entry.category == "operation":
            if _dfgsb_visible_opcode(entry.opcode):
                filtered.append(entry)
            continue
        region = region_by_id.get(entry.region_id)
        if region is None:
            continue
        producer = producer_by_region[region.id].get(entry.node_id)
        if producer is None or not _dfgsb_visible_opcode(producer.opcode):
            continue
        consumers = _dfgsb_value_consumers(design, region, producer, helper)
        if any(_dfgsb_visible_opcode(consumer.node.opcode) for consumer in consumers):
            filtered.append(entry)
    return filtered


def _conflict_edges(entries: list[_BoundOccurrence]) -> list[tuple[str, str]]:
    conflicts = interval_conflicts(entries, lambda entry: (entry.start, entry.end), key=lambda entry: entry.display_id)
    edges: list[tuple[str, str]] = []
    for source_id, targets in sorted(conflicts.items()):
        for target_id in sorted(targets):
            if source_id < target_id:
                edges.append((source_id, target_id))
    return edges


def _compatibility_edges(entries: list[_BoundOccurrence]) -> list[tuple[str, str]]:
    conflict_set = {tuple(sorted(edge)) for edge in _conflict_edges(entries)}
    edges: list[tuple[str, str]] = []
    for left, right in combinations(entries, 2):
        edge = tuple(sorted((left.display_id, right.display_id)))
        if edge in conflict_set:
            continue
        edges.append(edge)
    edges.sort()
    return edges


def _render_dfgsb_text(entries: list[_BoundOccurrence], *, compact: bool) -> list[str]:
    rows = _render_dfgsb_rows(entries, compact=compact)
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    rendered: list[str] = []
    for row_index, row in enumerate(rows):
        rendered.append(" | ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)))
        if row_index == 0:
            rendered.append("-+-".join("-" * width for width in widths))
    return rendered


def _render_dfgsb_html_table(layout: _DFGSBTableLayout, entries: list[_BoundOccurrence], *, compact: bool) -> str:
    if not entries:
        return '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD>empty</TD></TR></TABLE>'
    operation_resources = sorted({entry.bind for entry in entries if entry.category == "operation"})
    register_resources = sorted({entry.bind for entry in entries if entry.category == "register"})
    resource_ids = [*operation_resources, *register_resources]
    resource_colors = _resource_color_map(resource_ids)
    min_time = min(entry.start for entry in entries)
    max_time = max(entry.end for entry in entries)
    cells = ['<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">']
    cells.append(
        "<TR>"
        + '<TD BGCOLOR="#e8e8e8"><B>slot</B></TD>'
        + "".join(f'<TD BGCOLOR="#e8e8e8"><B>{_escape_html(resource_id)}</B></TD>' for resource_id in resource_ids)
        + "</TR>"
    )

    for time_step in range(min_time, max_time + 1):
        cells.append("<TR>")
        cells.append(f'<TD BGCOLOR="#ffffff"><B>cc {time_step}</B></TD>')
        for resource_id in resource_ids:
            if resource_id not in operation_resources:
                cells.append('<TD BGCOLOR="#ffffff"></TD>')
                continue
            occupants = [
                entry
                for entry in entries
                if entry.category == "operation" and entry.bind == resource_id and entry.start <= time_step <= entry.end
            ]
            cells.append(_dfgsb_html_cell(occupants, time_step, resource_colors.get(resource_id, "#ffffff"), compact=compact, is_register=False, layout=layout))
        cells.append("</TR>")

        reg_occupants_exist = any(
            entry.category == "register" and entry.start <= time_step + 1 <= entry.end
            for entry in entries
        )
        if not reg_occupants_exist:
            continue
        cells.append("<TR>")
        cells.append(f'<TD BGCOLOR="#f7f7f7"><B>reg {time_step + 1}</B></TD>')
        for resource_id in resource_ids:
            if resource_id not in register_resources:
                cells.append('<TD BGCOLOR="#f7f7f7"></TD>')
                continue
            occupants = [
                entry
                for entry in entries
                if entry.category == "register" and entry.bind == resource_id and entry.start <= time_step + 1 <= entry.end
            ]
            cells.append(
                _dfgsb_html_cell(
                    occupants,
                    time_step + 1,
                    resource_colors.get(resource_id, "#f7f7f7"),
                    compact=compact,
                    is_register=True,
                    layout=layout,
                )
            )
        cells.append("</TR>")
    cells.append("</TABLE>")
    return "".join(cells)


def _dfgsb_html_cell(
    occupants: list[_BoundOccurrence],
    time_step: int,
    background: str,
    *,
    compact: bool,
    is_register: bool,
    layout: _DFGSBTableLayout,
) -> str:
    if not occupants:
        return f'<TD BGCOLOR="{background}"></TD>'
    text = ", ".join(_entry_body(entry, compact=compact) for entry in occupants)
    ports: list[str] = []
    if len(occupants) == 1:
        entry = occupants[0]
        if is_register:
            port = _port_name(f"reg_{entry.display_id}_{time_step}")
            layout.reg_ports[(entry.display_id, time_step)] = port
            ports.append(f' PORT="{port}"')
        else:
            if entry.start == time_step:
                port = _port_name(f"in_{entry.display_id}_{time_step}")
                layout.op_in_ports.setdefault(entry.display_id, port)
                ports.append(f' PORT="{port}"')
            if entry.end == time_step and entry.start != time_step:
                port = _port_name(f"out_{entry.display_id}_{time_step}")
                layout.op_out_ports.setdefault(entry.display_id, port)
                ports = [f' PORT="{port}"']
            elif entry.end == time_step:
                port = layout.op_in_ports.get(entry.display_id, _port_name(f"io_{entry.display_id}_{time_step}"))
                layout.op_in_ports.setdefault(entry.display_id, port)
                layout.op_out_ports.setdefault(entry.display_id, port)
                ports = [f' PORT="{port}"']
    port_attr = "" if not ports else ports[-1]
    return f'<TD{port_attr} BGCOLOR="{background}">{_escape_html(text)}</TD>'


def _render_dfgsb_rows(entries: list[_BoundOccurrence], *, compact: bool) -> list[list[str]]:
    if not entries:
        return [["slot"], ["<empty>"]]
    operation_resources = sorted({entry.bind for entry in entries if entry.category == "operation"})
    register_resources = sorted({entry.bind for entry in entries if entry.category == "register"})
    resource_ids = [*operation_resources, *register_resources]
    if not resource_ids:
        return [["slot"], ["<empty>"]]

    min_time = min(entry.start for entry in entries)
    max_time = max(entry.end for entry in entries)
    rows: list[list[str]] = [["slot", *resource_ids]]
    for time_step in range(min_time, max_time + 1):
        operation_row = [f"cc {time_step}"]
        for resource_id in resource_ids:
            if resource_id not in operation_resources:
                operation_row.append("")
                continue
            occupants = [
                _entry_body(entry, compact=compact)
                for entry in entries
                if entry.category == "operation" and entry.bind == resource_id and entry.start <= time_step <= entry.end
            ]
            operation_row.append(", ".join(sorted(occupants)))
        if any(cell for cell in operation_row[1:]):
            rows.append(operation_row)

        register_row = [f"reg {time_step + 1}"]
        for resource_id in resource_ids:
            if resource_id not in register_resources:
                register_row.append("")
                continue
            occupants = [
                _entry_body(entry, compact=compact)
                for entry in entries
                if entry.category == "register" and entry.bind == resource_id and entry.start <= time_step + 1 <= entry.end
            ]
            register_row.append(", ".join(sorted(occupants)))
        if any(cell for cell in register_row[1:]):
            rows.append(register_row)
    if len(rows) == 1:
        rows.append(["<empty>", *([""] * len(resource_ids))])
    return rows


def _render_dfgsb_dot_scope(
    *,
    cluster_id: str,
    label: str,
    entries: list[_BoundOccurrence],
    source_specs: list[_DFGSBSourceSpec],
    output_specs: list[_DFGSBOutputSpec],
    compact: bool,
) -> tuple[list[str], _DFGSBDotLayout]:
    lines = [
        f'  subgraph "cluster_{cluster_id}" {{',
        f'    label="{_escape_dot(label)}";',
        "    color=gray70;",
    ]
    layout = _DFGSBDotLayout(cluster_id=cluster_id, op_nodes={}, reg_nodes={}, source_nodes={}, output_nodes={})
    if not entries:
        lines.append(f'    "{cluster_id}_empty" [label="empty", shape=plaintext, style=""];')
        lines.append("  }")
        return lines, layout

    lanes = _dfgsb_lane_order(entries)
    resource_colors = _resource_color_map(lanes)
    lane_headers = []
    for index, lane in enumerate(lanes):
        del lane
        header_id = f"{cluster_id}_lane_{index}"
        lane_headers.append(header_id)
        lines.append(
            f'    "{header_id}" [label="", shape=point, width=0.01, height=0.01, style=invis];'
        )
    lines.append("    { rank=same; " + " ".join(f'"{node_id}";' for node_id in lane_headers) + " }")
    for left_id, right_id in zip(lane_headers, lane_headers[1:]):
        lines.append(f'    "{left_id}" -> "{right_id}" [style=invis, weight=50];')

    if source_specs:
        source_node_ids: list[str] = []
        for index, source_spec in enumerate(source_specs):
            source_id = f"{cluster_id}_src_{index}"
            source_node_ids.append(source_id)
            layout.source_nodes[source_spec.key] = source_id
            lines.append(
                f'    "{source_id}" [label="{_escape_dot_label(source_spec.label)}", {_dfgsb_source_style_attrs(source_spec.kind)}];'
            )
        lines.append("    { rank=same; " + " ".join(f'"{node_id}";' for node_id in source_node_ids) + " }")
        for left_id, right_id in zip(source_node_ids, source_node_ids[1:]):
            lines.append(f'    "{left_id}" -> "{right_id}" [style=invis, weight=30];')

    row_nodes_by_lane: dict[str, list[str]] = {lane: [] for lane in lanes}
    row_order = _dfgsb_row_order(entries)
    for row_index, (row_kind, time_step) in enumerate(row_order):
        label_id = f"{cluster_id}_{row_kind}_{time_step}_label"
        label_text = f"cc {time_step}" if row_kind == "op" else f"reg {time_step}"
        lines.append(f'    "{label_id}" [label="{_escape_dot(label_text)}", shape=plaintext, style=""];')
        row_nodes: list[str] = [label_id]
        for lane_index, lane in enumerate(lanes):
            occupants = _dfgsb_row_lane_entries(entries, row_kind=row_kind, time_step=time_step, lane=lane)
            if occupants:
                node_id, node_lines = _render_dfgsb_dot_occurrence(
                    cluster_id=cluster_id,
                    lane=lane,
                    lane_index=lane_index,
                    row_kind=row_kind,
                    time_step=time_step,
                    occupants=occupants,
                    compact=compact,
                    fillcolor=resource_colors[lane],
                    layout=layout,
                )
                lines.extend(node_lines)
            else:
                node_id = f"{cluster_id}_{row_kind}_{time_step}_{lane_index}_empty"
                lines.append(f'    "{node_id}" [label="", shape=point, width=0.01, height=0.01, style=invis];')
            row_nodes.append(node_id)
            row_nodes_by_lane[lane].append(node_id)
        lines.append("    { rank=same; " + " ".join(f'"{node_id}";' for node_id in row_nodes) + " }")
        for left_id, right_id in zip(row_nodes, row_nodes[1:]):
            lines.append(f'    "{left_id}" -> "{right_id}" [style=invis, weight=40];')
    for lane, lane_nodes in row_nodes_by_lane.items():
        del lane
        for top_id, bottom_id in zip(lane_nodes, lane_nodes[1:]):
            lines.append(f'    "{top_id}" -> "{bottom_id}" [style=invis, weight=20];')
    if output_specs:
        output_node_ids: list[str] = []
        for index, output_spec in enumerate(output_specs):
            output_id = f"{cluster_id}_out_{index}"
            output_node_ids.append(output_id)
            layout.output_nodes[output_spec.key] = output_id
            lines.append(
                f'    "{output_id}" [label="{_escape_dot_label(output_spec.label)}", {_dfgsb_source_style_attrs("input")}];'
            )
        lines.append("    { rank=same; " + " ".join(f'"{node_id}";' for node_id in output_node_ids) + " }")
        for left_id, right_id in zip(output_node_ids, output_node_ids[1:]):
            lines.append(f'    "{left_id}" -> "{right_id}" [style=invis, weight=30];')
    lines.append("  }")
    return lines, layout


def _render_dfgsb_dot_occurrence(
    *,
    cluster_id: str,
    lane: str,
    lane_index: int,
    row_kind: str,
    time_step: int,
    occupants: list[_BoundOccurrence],
    compact: bool,
    fillcolor: str,
    layout: _DFGSBDotLayout,
) -> tuple[str, list[str]]:
    node_id = f"{cluster_id}_{row_kind}_{time_step}_{lane_index}"
    if len(occupants) == 1:
        entry = occupants[0]
        if entry.category == "operation":
            opcode = _compact_opcode(entry.opcode) if compact else entry.opcode
            label = f"{entry.display_id}\\n{opcode}\\n{_entry_bind_label(entry)}"
            shape = "ellipse"
            layout.op_nodes[(entry.region_id, entry.display_id)] = node_id
        else:
            label = _entry_bind_label(entry)
            shape = "box"
            layout.reg_nodes[(entry.region_id, entry.display_id, time_step)] = node_id
        return node_id, [f'    "{node_id}" [label="{_escape_dot_label(label)}", shape={shape}, fillcolor="{fillcolor}"];']

    joined = "\\n".join(_entry_body(entry, compact=compact) for entry in occupants)
    return node_id, [f'    "{node_id}" [label="{_escape_dot_label(joined)}", shape=box, fillcolor="{fillcolor}"];']


def _dot_dfgsb_region_edges(
    design: UHIRDesign,
    layouts: dict[str, _DFGSBDotLayout],
    grouped_entries: dict[str, list[_BoundOccurrence]],
) -> list[str]:
    helper = OperationBinderBase()
    lines: list[str] = []
    for region in design.regions:
        layout = layouts.get(region.id)
        if layout is None:
            continue
        lines.extend(_dot_dfgsb_region_source_edges(design, region, layout))
        lines.extend(_dot_dfgsb_region_output_edges(design, region, layout))
        lines.extend(_dot_dfgsb_region_data_edges(region, layout, helper))
        lines.extend(
            _dot_dfgsb_region_register_edges(
                design,
                region,
                layout,
                layouts,
                grouped_entries.get(region.id, []),
                helper,
            )
        )
    return lines


def _dot_dfgsb_region_source_edges(design: UHIRDesign, region: UHIRRegion, layout: _DFGSBDotLayout) -> list[str]:
    lines: list[str] = []
    seen_edges: set[tuple[str, str]] = set()
    for node in region.nodes:
        if not _dfgsb_visible_opcode(node.opcode):
            continue
        target_id = layout.op_nodes.get((region.id, node.id))
        if target_id is None:
            continue
        for operand in node.operands:
            source_spec = _dfgsb_operand_source_spec(design, operand)
            if source_spec is None:
                continue
            source_id = layout.source_nodes.get(source_spec.key)
            if source_id is None:
                continue
            edge_key = (source_id, target_id)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            lines.append(f'  "{source_id}" -> "{target_id}" [color="#444444", penwidth=1.1];')
    return lines


def _dot_dfgsb_region_output_edges(design: UHIRDesign, region: UHIRRegion, layout: _DFGSBDotLayout) -> list[str]:
    if region.parent is not None:
        return []
    output_specs = _collect_dfgsb_output_specs(design)
    if not output_specs:
        return []
    lines: list[str] = []
    for node in region.nodes:
        if node.opcode != "ret":
            continue
        source_id = layout.op_nodes.get((region.id, node.id))
        if source_id is None:
            continue
        for output_spec in output_specs:
            if output_spec.region_id != region.id or output_spec.node_id != node.id:
                continue
            target_id = layout.output_nodes.get(output_spec.key)
            if target_id is None:
                continue
            lines.append(f'  "{source_id}" -> "{target_id}" [color="#444444", penwidth=1.3];')
    return lines


def _dot_dfgsb_region_data_edges(region: UHIRRegion, layout: _DFGSBDotLayout, helper: OperationBinderBase) -> list[str]:
    lines: list[str] = []
    bound_producers = _bound_producer_node_ids(region)
    for node in region.nodes:
        if not _dfgsb_visible_opcode(node.opcode) or node.id in bound_producers:
            continue
        source_id = layout.op_nodes.get((region.id, node.id))
        if source_id is None:
            continue
        for target in _dfgsb_visible_targets(region, node.id, helper):
            target_id = layout.op_nodes.get((region.id, target.id))
            if target_id is None:
                continue
            lines.append(f'  "{source_id}" -> "{target_id}" [color="#444444", penwidth=1.3];')
    return lines


def _dot_dfgsb_region_register_edges(
    design: UHIRDesign,
    region: UHIRRegion,
    layout: _DFGSBDotLayout,
    layouts: dict[str, _DFGSBDotLayout],
    local_entries: list[_BoundOccurrence],
    helper: OperationBinderBase,
) -> list[str]:
    lines: list[str] = []
    seen_edges: set[tuple[str, str, str]] = set()
    producer_by_name = _producer_node_by_name(region)
    local_reg_entries = [
        entry
        for entry in local_entries
        if entry.category == "register"
    ]
    for binding in region.value_bindings:
        producer = producer_by_name.get(binding.producer)
        if producer is None or not _dfgsb_visible_opcode(producer.opcode):
            continue
        value_label = binding.producer
        producer_id = layout.op_nodes.get((region.id, producer.id))
        if producer_id is None:
            continue
        local_consumers = [
            consumer
            for consumer in helper.get_local_value_consumers(region, producer)
            if _dfgsb_visible_opcode(consumer.node.opcode)
        ]
        external_consumers = [
            consumer
            for consumer in _dfgsb_value_consumers(design, region, producer, helper)
            if consumer.region.id != region.id and _dfgsb_visible_opcode(consumer.node.opcode)
        ]
        matching_entries = [
            entry
            for entry in local_reg_entries
            if entry.node_id in helper.get_value_names(region, producer)
        ]
        if not matching_entries:
            producer_id = layout.op_nodes.get((region.id, producer.id))
            if producer_id is not None:
                for target in _dfgsb_visible_targets(region, producer.id, helper):
                    target_id = layout.op_nodes.get((region.id, target.id))
                    if target_id is None:
                        continue
                    edge_key = (producer_id, target_id, "hidden")
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    lines.append(
                        f'  "{producer_id}" -> "{target_id}" [color="#1f78b4", penwidth=1.3{_edge_label_attr(value_label)}];'
                    )
        for reg_entry in matching_entries:
            first_reg_id = layout.reg_nodes.get((region.id, reg_entry.display_id, reg_entry.start))
            if first_reg_id is not None:
                edge_key = (producer_id, first_reg_id, "local")
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    lines.append(
                        f'  "{producer_id}" -> "{first_reg_id}" [color="#1f78b4", penwidth=1.3{_edge_label_attr(value_label)}];'
                    )
            for step in range(reg_entry.start, reg_entry.end):
                left_id = layout.reg_nodes.get((region.id, reg_entry.display_id, step))
                right_id = layout.reg_nodes.get((region.id, reg_entry.display_id, step + 1))
                if left_id is None or right_id is None:
                    continue
                edge_key = (left_id, right_id, "hold")
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                lines.append(
                    f'  "{left_id}" -> "{right_id}" [color="#1f78b4", penwidth=1.1{_edge_label_attr(value_label)}];'
                )
            for consumer in local_consumers:
                target_layout = layouts.get(consumer.region.id)
                if target_layout is None:
                    continue
                target_row = helper.get_node_interval(consumer.region, consumer.node)[0] + consumer.start_shift
                consumer_id = target_layout.op_nodes.get((consumer.region.id, consumer.node.id))
                if consumer_id is None:
                    continue
                if target_row < reg_entry.start:
                    edge_key = (producer_id, consumer_id, "bypass")
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        lines.append(
                            f'  "{producer_id}" -> "{consumer_id}" [color="#1f78b4", penwidth=1.3{_edge_label_attr(value_label)}];'
                        )
                    continue
                reg_id = layout.reg_nodes.get((region.id, reg_entry.display_id, target_row))
                if reg_id is None:
                    continue
                edge_key = (reg_id, consumer_id, "consume")
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                lines.append(
                    f'  "{reg_id}" -> "{consumer_id}" [color="#1f78b4", penwidth=1.3{_edge_label_attr(value_label)}];'
                )
            exit_reg_id = layout.reg_nodes.get((region.id, reg_entry.display_id, reg_entry.end)) or first_reg_id or producer_id
            for consumer in external_consumers:
                target_layout = layouts.get(consumer.region.id)
                if target_layout is None:
                    continue
                consumer_id = target_layout.op_nodes.get((consumer.region.id, consumer.node.id))
                if consumer_id is None:
                    continue
                edge_key = (exit_reg_id, consumer_id, "cross")
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                lines.append(
                    f'  "{exit_reg_id}" -> "{consumer_id}" [color="#1f78b4", penwidth=1.1, style=dashed{_edge_label_attr(value_label)}];'
                )
    return lines


def _dot_dfgsb_unroll_edges(design: UHIRDesign, layout: _DFGSBDotLayout, entries: list[_BoundOccurrence]) -> list[str]:
    helper = OperationBinderBase()
    lines: list[str] = []
    seen_edges: set[tuple[str, str, str]] = set()
    op_index: dict[tuple[str, str, str], str] = {}
    op_entries_by_node: dict[tuple[str, str], list[_BoundOccurrence]] = {}
    reg_entries_by_node: dict[tuple[str, str], list[_BoundOccurrence]] = {}
    producer_ref_by_name: dict[str, tuple[UHIRRegion, UHIRNode]] = {}

    for region in design.regions:
        for node in region.nodes:
            for value_name in helper.get_value_names(region, node):
                producer_ref_by_name.setdefault(value_name, (region, node))

    for entry in entries:
        if entry.category == "operation":
            op_node_id = layout.op_nodes.get((entry.region_id, entry.display_id))
            if op_node_id is None:
                continue
            suffix = _display_suffix(entry.node_id, entry.display_id)
            op_index[(entry.region_id, entry.node_id, suffix)] = op_node_id
            op_entries_by_node.setdefault((entry.region_id, entry.node_id), []).append(entry)
        elif entry.category == "register":
            reg_entries_by_node.setdefault((entry.region_id, entry.node_id), []).append(entry)

    for node_entries in op_entries_by_node.values():
        node_entries.sort(key=lambda entry: (entry.start, entry.end, entry.display_id))
    for node_entries in reg_entries_by_node.values():
        node_entries.sort(key=lambda entry: (entry.start, entry.end, entry.display_id))

    for region in design.regions:
        for node in region.nodes:
            if not _dfgsb_visible_opcode(node.opcode):
                continue
            for entry in op_entries_by_node.get((region.id, node.id), []):
                target_id = layout.op_nodes.get((entry.region_id, entry.display_id))
                if target_id is None:
                    continue
                for operand_index, operand in enumerate(node.operands):
                    source_spec = _dfgsb_operand_source_spec(design, operand)
                    if source_spec is None:
                        continue
                    source_id = _ensure_dfgsb_unroll_source_node(
                        layout=layout,
                        lines=lines,
                        source_spec=source_spec,
                        target_entry=entry,
                        operand_index=operand_index,
                    )
                    edge_key = (source_id, target_id, "source")
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    lines.append(f'  "{source_id}" -> "{target_id}" [color="#444444", penwidth=1.1];')

    output_specs = _collect_dfgsb_output_specs(design)
    if output_specs:
        for region in design.regions:
            if region.parent is not None:
                continue
            for node in region.nodes:
                if node.opcode != "ret":
                    continue
                for entry in op_entries_by_node.get((region.id, node.id), []):
                    source_id = layout.op_nodes.get((entry.region_id, entry.display_id))
                    if source_id is None:
                        continue
                    for output_spec in output_specs:
                        if output_spec.region_id != region.id or output_spec.node_id != node.id:
                            continue
                        target_id = layout.output_nodes.get(output_spec.key)
                        if target_id is None:
                            continue
                        edge_key = (source_id, target_id, "output")
                        if edge_key in seen_edges:
                            continue
                        seen_edges.add(edge_key)
                        lines.append(f'  "{source_id}" -> "{target_id}" [color="#444444", penwidth=1.3];')

    for region in design.regions:
        bound_producers = _bound_producer_node_ids(region)
        for node in region.nodes:
            if not _dfgsb_visible_opcode(node.opcode) or node.id in bound_producers:
                continue
            for suffix in _suffixes_for_region_node(entries, region.id, node.id):
                source_id = op_index.get((region.id, node.id, suffix))
                if source_id is None:
                    continue
                for target in _dfgsb_visible_targets(region, node.id, helper):
                    target_id = op_index.get((region.id, target.id, suffix))
                    if target_id is None:
                        continue
                    edge_key = (source_id, target_id, "data")
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    lines.append(f'  "{source_id}" -> "{target_id}" [color="#444444", penwidth=1.3];')

        producer_by_name = _producer_node_by_name(region)
        for binding in region.value_bindings:
            producer = producer_by_name.get(binding.producer)
            if producer is None or not _dfgsb_visible_opcode(producer.opcode):
                continue
            value_label = binding.producer
            producer_names = helper.get_value_names(region, producer)
            producer_occurrences = op_entries_by_node.get((region.id, producer.id), [])
            register_entries = [
                entry
                for name in producer_names
                for entry in reg_entries_by_node.get((region.id, name), [])
            ]
            if not producer_occurrences:
                continue
            consumers = [
                consumer
                for consumer in _dfgsb_value_consumers(design, region, producer, helper)
                if _dfgsb_visible_opcode(consumer.node.opcode)
            ]
            if not register_entries:
                for suffix in _suffixes_for_region_node(entries, region.id, producer.id):
                    source_id = op_index.get((region.id, producer.id, suffix))
                    if source_id is None:
                        continue
                    for target in _dfgsb_visible_targets(region, producer.id, helper):
                        target_id = op_index.get((region.id, target.id, suffix))
                        if target_id is None:
                            continue
                        edge_key = (source_id, target_id, "hidden")
                        if edge_key in seen_edges:
                            continue
                        seen_edges.add(edge_key)
                        lines.append(
                            f'  "{source_id}" -> "{target_id}" [color="#1f78b4", penwidth=1.3{_edge_label_attr(value_label)}];'
                        )
                continue
            for reg_entry in register_entries:
                matched_consumer_keys: set[tuple[str, str]] = set()
                producer_entry = next(
                    (entry for entry in reversed(producer_occurrences) if entry.end < reg_entry.start),
                    None,
                )
                if producer_entry is None:
                    producer_entry = next(
                        (entry for entry in reversed(producer_occurrences) if entry.end <= reg_entry.start),
                        None,
                    )
                if producer_entry is None:
                    continue
                producer_id = layout.op_nodes.get((producer_entry.region_id, producer_entry.display_id))
                first_reg_id = layout.reg_nodes.get((reg_entry.region_id, reg_entry.display_id, reg_entry.start))
                if producer_id is None or first_reg_id is None:
                    continue
                edge_key = (producer_id, first_reg_id, "write")
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    lines.append(
                        f'  "{producer_id}" -> "{first_reg_id}" [color="#1f78b4", penwidth=1.3{_edge_label_attr(value_label)}];'
                    )
                if producer.opcode == "phi":
                    alias_suffix = _display_suffix(reg_entry.node_id, reg_entry.display_id)
                    for operand_name in producer.operands:
                        alias_ref = producer_ref_by_name.get(operand_name)
                        if alias_ref is None:
                            continue
                        alias_region, alias_node = alias_ref
                        if not _dfgsb_visible_opcode(alias_node.opcode):
                            continue
                        alias_entries = op_entries_by_node.get((alias_region.id, alias_node.id), [])
                        alias_entry = next(
                            (
                                candidate
                                for candidate in alias_entries
                                if _display_suffix(alias_node.id, candidate.display_id) == alias_suffix
                                and reg_entry.start <= candidate.end + 1 <= reg_entry.end
                            ),
                            None,
                        )
                        if alias_entry is None:
                            continue
                        alias_op_id = layout.op_nodes.get((alias_entry.region_id, alias_entry.display_id))
                        alias_reg_id = layout.reg_nodes.get((reg_entry.region_id, reg_entry.display_id, alias_entry.end + 1))
                        if alias_op_id is None or alias_reg_id is None:
                            continue
                        edge_key = (alias_op_id, alias_reg_id, "alias_write")
                        if edge_key in seen_edges:
                            continue
                        seen_edges.add(edge_key)
                        lines.append(
                            f'  "{alias_op_id}" -> "{alias_reg_id}" [color="#1f78b4", penwidth=1.3{_edge_label_attr(value_label)}];'
                        )
                for step in range(reg_entry.start, reg_entry.end):
                    left_id = layout.reg_nodes.get((reg_entry.region_id, reg_entry.display_id, step))
                    right_id = layout.reg_nodes.get((reg_entry.region_id, reg_entry.display_id, step + 1))
                    if left_id is None or right_id is None:
                        continue
                    edge_key = (left_id, right_id, "hold")
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    lines.append(
                        f'  "{left_id}" -> "{right_id}" [color="#1f78b4", penwidth=1.1{_edge_label_attr(value_label)}];'
                    )
                for consumer in consumers:
                    for consumer_entry in op_entries_by_node.get((consumer.region.id, consumer.node.id), []):
                        if not (reg_entry.start <= consumer_entry.start <= reg_entry.end):
                            continue
                        reg_id = layout.reg_nodes.get((reg_entry.region_id, reg_entry.display_id, consumer_entry.start))
                        consumer_id = layout.op_nodes.get((consumer_entry.region_id, consumer_entry.display_id))
                        if reg_id is None or consumer_id is None:
                            continue
                        edge_key = (reg_id, consumer_id, "consume")
                        if edge_key in seen_edges:
                            continue
                        matched_consumer_keys.add((consumer.region.id, consumer.node.id))
                        seen_edges.add(edge_key)
                        lines.append(
                            f'  "{reg_id}" -> "{consumer_id}" [color="#1f78b4", penwidth=1.3{_edge_label_attr(value_label)}];'
                        )
                exit_reg_id = layout.reg_nodes.get((reg_entry.region_id, reg_entry.display_id, reg_entry.end)) or first_reg_id
                for consumer in consumers:
                    if consumer.node.opcode != "phi":
                        continue
                    if (consumer.region.id, consumer.node.id) in matched_consumer_keys:
                        continue
                    for alias_consumer in _dfgsb_value_consumers(design, consumer.region, consumer.node, helper):
                        if not _dfgsb_visible_opcode(alias_consumer.node.opcode):
                            continue
                        for consumer_entry in op_entries_by_node.get((alias_consumer.region.id, alias_consumer.node.id), []):
                            if consumer_entry.start < reg_entry.start:
                                continue
                            consumer_id = layout.op_nodes.get((consumer_entry.region_id, consumer_entry.display_id))
                            if exit_reg_id is None or consumer_id is None:
                                continue
                            edge_key = (exit_reg_id, consumer_id, "alias_consume")
                            if edge_key in seen_edges:
                                continue
                            seen_edges.add(edge_key)
                            lines.append(
                                f'  "{exit_reg_id}" -> "{consumer_id}" [color="#1f78b4", penwidth=1.3{_edge_label_attr(value_label)}];'
                            )
    return lines


def _render_trp_text(entries: list[_BoundOccurrence], *, compact: bool) -> list[str]:
    if not entries:
        return ["<empty>"]
    resource_ids = sorted({entry.bind for entry in entries})
    min_time = min(entry.start for entry in entries)
    max_time = max(entry.end for entry in entries)
    rows: list[list[str]] = [["t", *resource_ids]]
    for time_step in range(min_time, max_time + 1):
        row = [str(time_step)]
        for resource_id in resource_ids:
            occupants = [
                _entry_body(entry, compact=compact)
                for entry in entries
                if entry.bind == resource_id and entry.start <= time_step <= entry.end
            ]
            row.append(", ".join(occupants))
        rows.append(row)
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    rendered: list[str] = []
    for row_index, row in enumerate(rows):
        rendered.append(" | ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)))
        if row_index == 0:
            rendered.append("-+-".join("-" * width for width in widths))
    return rendered


def _render_trp_html_table(entries: list[_BoundOccurrence], *, compact: bool) -> str:
    if not entries:
        return '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD>empty</TD></TR></TABLE>'
    resource_ids = sorted({entry.bind for entry in entries})
    min_time = min(entry.start for entry in entries)
    max_time = max(entry.end for entry in entries)
    cells = ['<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">']
    cells.append("<TR><TD><B>t</B></TD>" + "".join(f"<TD><B>{_escape_html(resource_id)}</B></TD>" for resource_id in resource_ids) + "</TR>")
    for time_step in range(min_time, max_time + 1):
        row = [f"<TR><TD>{time_step}</TD>"]
        for resource_id in resource_ids:
            occupants = [
                _entry_body(entry, compact=compact)
                for entry in entries
                if entry.bind == resource_id and entry.start <= time_step <= entry.end
            ]
            row.append(f"<TD>{_escape_html(', '.join(occupants))}</TD>")
        row.append("</TR>")
        cells.append("".join(row))
    cells.append("</TABLE>")
    return "".join(cells)


def _entry_label(entry: _BoundOccurrence, *, compact: bool) -> str:
    opcode = _compact_opcode(entry.opcode) if compact else entry.opcode
    return f"{entry.display_id} {opcode} {_entry_bind_label(entry)}"


def _entry_body(entry: _BoundOccurrence, *, compact: bool) -> str:
    if entry.category == "register":
        if entry.scope_id == "global":
            return entry.display_id.removeprefix("reg_")
        return entry.node_id
    opcode = _compact_opcode(entry.opcode) if compact else entry.opcode
    return f"{entry.display_id} {opcode}"


def _compact_opcode(opcode: str) -> str:
    return COMPACT_OPCODE_LABELS.get(opcode, opcode)


def _node_children(node: UHIRNode) -> list[str]:
    children: list[str] = []
    for key in ("child", "true_child", "false_child"):
        child_id = node.attributes.get(key)
        if isinstance(child_id, str) and child_id:
            children.append(child_id)
    return children


def _dfgsb_visible_opcode(opcode: str) -> bool:
    return opcode not in {"nop", "branch", "loop"}


def _display_suffix(node_id: str, display_id: str) -> str:
    if display_id.startswith("reg_"):
        display_id = display_id.removeprefix("reg_")
    if display_id == node_id:
        return ""
    if display_id.startswith(node_id):
        suffix = display_id[len(node_id) :]
        if suffix.startswith("@"):
            cut = suffix.find("_")
            if cut != -1:
                return suffix[:cut]
        return suffix
    return ""


def _suffixes_for_region_node(entries: list[_BoundOccurrence], region_id: str, node_id: str) -> list[str]:
    suffixes = {
        _display_suffix(node_id, entry.display_id)
        for entry in entries
        if entry.category == "operation" and entry.region_id == region_id and entry.node_id == node_id
    }
    return sorted(suffixes)


def _dfgsb_lane_order(entries: list[_BoundOccurrence]) -> list[str]:
    operation_lanes = sorted({_dfgsb_lane_key(entry) for entry in entries if entry.category == "operation"})
    register_lanes = sorted({entry.bind for entry in entries if entry.category == "register"})
    return [*operation_lanes, *register_lanes]


def _dfgsb_row_order(entries: list[_BoundOccurrence]) -> list[tuple[str, int]]:
    if not entries:
        return []
    min_time = min(entry.start for entry in entries)
    max_time = max(entry.end for entry in entries)
    rows: list[tuple[str, int]] = []
    for time_step in range(min_time, max_time + 1):
        if any(entry.category == "operation" and entry.start <= time_step <= entry.end for entry in entries):
            rows.append(("op", time_step))
        register_time = time_step + 1
        if any(entry.category == "register" and entry.start <= register_time <= entry.end for entry in entries):
            rows.append(("reg", register_time))
    return rows


def _dfgsb_row_lane_entries(
    entries: list[_BoundOccurrence],
    *,
    row_kind: str,
    time_step: int,
    lane: str,
) -> list[_BoundOccurrence]:
    if row_kind == "op":
        return [
            entry
            for entry in entries
            if entry.category == "operation" and _dfgsb_lane_key(entry) == lane and entry.start <= time_step <= entry.end
        ]
    return [
        entry
        for entry in entries
        if entry.category == "register" and entry.bind == lane and entry.start <= time_step <= entry.end
    ]


def _dfgsb_lane_key(entry: _BoundOccurrence) -> str:
    if entry.category == "operation" and entry.class_name == "CTRL" and entry.bind == "ctrl":
        return f"ctrl:{entry.node_id}"
    return entry.bind


def _dfgsb_source_style_attrs(kind: str) -> str:
    if kind == "input":
        return 'shape=box, fillcolor="#ffffff", color="#222222", style="filled,bold", penwidth=2'
    return 'shape=box, fillcolor="#eeeeee", color="#999999", style="filled,dashed", penwidth=1.3'


def _collect_dfgsb_source_specs_from_nodes(design: UHIRDesign, nodes: list[UHIRNode]) -> list[_DFGSBSourceSpec]:
    source_specs: dict[str, _DFGSBSourceSpec] = {}
    for node in nodes:
        if not _dfgsb_visible_opcode(node.opcode):
            continue
        for operand in node.operands:
            source_spec = _dfgsb_operand_source_spec(design, operand)
            if source_spec is None:
                continue
            source_specs.setdefault(source_spec.key, source_spec)
    return sorted(source_specs.values(), key=lambda spec: (spec.kind, spec.label))


def _collect_dfgsb_output_specs(design: UHIRDesign) -> list[_DFGSBOutputSpec]:
    specs: list[_DFGSBOutputSpec] = []
    for region in design.regions:
        if region.parent is not None:
            continue
        for node in region.nodes:
            if node.opcode != "ret":
                continue
            label = node.operands[0] if node.operands else (design.outputs[0].name if design.outputs else node.id)
            specs.append(
                _DFGSBOutputSpec(
                    key=f"output:{region.id}:{node.id}",
                    label=label,
                    region_id=region.id,
                    node_id=node.id,
                )
            )
    return specs


def _dfgsb_operand_source_spec(design: UHIRDesign, operand: str) -> _DFGSBSourceSpec | None:
    if operand in {port.name for port in design.inputs}:
        return _DFGSBSourceSpec(key=f"input:{operand}", label=operand, kind="input")
    if operand in {const_decl.name for const_decl in design.constants}:
        return _DFGSBSourceSpec(key=f"const:{operand}", label=operand, kind="constant")
    literal_head, separator, _literal_tail = operand.partition(":")
    if separator and literal_head and literal_head[0] in "-0123456789":
        return _DFGSBSourceSpec(key=f"const:{operand}", label=operand, kind="constant")
    return None


def _ensure_dfgsb_unroll_source_node(
    *,
    layout: _DFGSBDotLayout,
    lines: list[str],
    source_spec: _DFGSBSourceSpec,
    target_entry: _BoundOccurrence,
    operand_index: int,
) -> str:
    source_id = (
        f"{layout.cluster_id}_src_"
        f"{_port_name(target_entry.display_id)}_{target_entry.start}_{operand_index}"
    )
    if source_id in layout.source_nodes:
        return layout.source_nodes[source_id]
    layout.source_nodes[source_id] = source_id
    lines.append(
        f'  "{source_id}" [label="{_escape_dot_label(source_spec.label)}", {_dfgsb_source_style_attrs(source_spec.kind)}];'
    )
    return source_id


def _entry_bind_label(entry: _BoundOccurrence) -> str:
    if entry.class_name == "CTRL" and entry.bind.startswith("ctrl:"):
        return "ctrl"
    return entry.bind


def _producer_node_by_name(region: UHIRRegion) -> dict[str, UHIRNode]:
    producer_by_name = {node.id: node for node in region.nodes}
    for mapping in region.mappings:
        node = next((candidate for candidate in region.nodes if candidate.id == mapping.node_id), None)
        if node is not None:
            producer_by_name[mapping.source_id] = node
    return producer_by_name


def _matching_value_bindings(
    region: UHIRRegion,
    bindings_by_name: dict[str, list[UHIRValueBinding]],
    producer_names: set[str],
) -> list[UHIRValueBinding]:
    matched: list[UHIRValueBinding] = []
    seen: set[tuple[str, str, tuple[tuple[int, int], ...]]] = set()
    for name in sorted(producer_names):
        for binding in bindings_by_name.get(name, []):
            key = (binding.producer, binding.register, binding.live_intervals)
            if key in seen:
                continue
            seen.add(key)
            matched.append(binding)
    if matched:
        return matched
    for binding in region.value_bindings:
        if binding.producer in producer_names:
            matched.append(binding)
    return matched


def _bind_design_looks_flattened(design: UHIRDesign, helper: OperationBinderBase) -> bool:
    del helper
    for region in design.regions:
        for binding in region.value_bindings:
            if len(binding.live_intervals) > 1:
                return True
    return False


def _collect_hierarchical_unrolled_register_entries(
    design: UHIRDesign,
    entries: list[_BoundOccurrence],
) -> list[_BoundOccurrence]:
    helper = OperationBinderBase()
    resources_by_id = {resource.id: resource for resource in design.resources}
    op_entries_by_node: dict[tuple[str, str], list[_BoundOccurrence]] = {}
    for entry in entries:
        if entry.category != "operation":
            continue
        op_entries_by_node.setdefault((entry.region_id, entry.node_id), []).append(entry)
    for producer_entries in op_entries_by_node.values():
        producer_entries.sort(key=lambda entry: (entry.start, entry.end, entry.display_id))

    register_entries: list[_BoundOccurrence] = []
    for region in design.regions:
        producer_by_name = _producer_node_by_name(region)
        for binding_index, binding in enumerate(region.value_bindings):
            register_resource = resources_by_id.get(binding.register)
            if register_resource is None or register_resource.kind != "reg":
                continue
            producer = producer_by_name.get(binding.producer)
            if producer is None:
                continue
            producer_occurrences = op_entries_by_node.get((region.id, producer.id), [])
            if not producer_occurrences:
                continue
            consumer_refs = helper.get_value_consumers(design, region, producer)
            for occurrence_index, producer_entry in enumerate(producer_occurrences):
                live_start = producer_entry.end + 1
                next_start = (
                    producer_occurrences[occurrence_index + 1].start
                    if occurrence_index + 1 < len(producer_occurrences)
                    else None
                )
                live_end_candidates: list[int] = []
                for consumer in consumer_refs:
                    consumer_occurrences = op_entries_by_node.get((consumer.region.id, consumer.node.id), [])
                    consumer_ii = max(helper.get_node_ii(consumer.region, consumer.node), 1)
                    for consumer_entry in consumer_occurrences:
                        if consumer_entry.start < live_start:
                            continue
                        if next_start is not None and consumer_entry.start >= next_start:
                            continue
                        live_end_candidates.append(consumer_entry.start + consumer_ii - 1)
                if not live_end_candidates:
                    continue
                suffix = _display_suffix(producer.id, producer_entry.display_id)
                display_id = f"reg_{binding.producer}{suffix}"
                if binding_index:
                    display_id = f"{display_id}_b{binding_index}"
                register_entries.append(
                    _BoundOccurrence(
                        category="register",
                        scope_id="global",
                        region_id=region.id,
                        node_id=binding.producer,
                        display_id=display_id,
                        opcode="reg",
                        bind=binding.register,
                        class_name=register_resource.value,
                        start=live_start,
                        end=max(live_end_candidates),
                    )
                )
    return register_entries


def _collect_flattened_register_entries(design: UHIRDesign) -> list[_BoundOccurrence]:
    resources_by_id = {resource.id: resource for resource in design.resources}
    entries: list[_BoundOccurrence] = []
    for region in design.regions:
        for binding_index, binding in enumerate(region.value_bindings):
            register_resource = resources_by_id.get(binding.register)
            if register_resource is None or register_resource.kind != "reg":
                continue
            for interval_index, (start, end) in enumerate(binding.live_intervals):
                display_id = f"reg_{binding.producer}[t={start}]"
                if len(binding.live_intervals) > 1:
                    display_id = f"{display_id}_{interval_index}"
                if binding_index:
                    display_id = f"{display_id}_b{binding_index}"
                entries.append(
                    _BoundOccurrence(
                        category="register",
                        scope_id="global",
                        region_id=region.id,
                        node_id=binding.producer,
                        display_id=display_id,
                        opcode="reg",
                        bind=binding.register,
                        class_name=register_resource.value,
                        start=start,
                        end=end,
                    )
                )
    return entries


def _edge_label_attr(text: str) -> str:
    return f', label="{_escape_dot_label(text)}", fontcolor="#1f78b4"'


def _dfgsb_value_consumers(
    design: UHIRDesign,
    region: UHIRRegion,
    producer: UHIRNode,
    helper: OperationBinderBase,
) -> list[ValueConsumerRef]:
    consumers = list(helper.get_value_consumers(design, region, producer))
    if region.parent is None:
        return consumers
    parent_region = design.get_region(region.parent)
    if parent_region is None:
        return consumers
    consumer_names = helper.get_value_names(region, producer)
    seen = {(consumer.region.id, consumer.node.id, consumer.start_shift) for consumer in consumers}
    for parent_node in parent_region.nodes:
        if parent_node.opcode in {"phi", "nop"}:
            continue
        if not any(operand in consumer_names for operand in parent_node.operands):
            continue
        key = (parent_region.id, parent_node.id, 0)
        if key in seen:
            continue
        seen.add(key)
        consumers.append(ValueConsumerRef(parent_region, parent_node, 0))
    return consumers


def _dfgsb_visible_targets(region: UHIRRegion, source_id: str, helper: OperationBinderBase) -> list[UHIRNode]:
    node_by_id = {node.id: node for node in region.nodes}
    outgoing: dict[str, list[str]] = {}
    for edge in helper.iter_region_data_edges(region):
        outgoing.setdefault(edge.source, []).append(edge.target)
    visible: list[UHIRNode] = []
    seen_hidden: set[str] = set()
    worklist = list(outgoing.get(source_id, ()))
    while worklist:
        target_id = worklist.pop(0)
        target = node_by_id.get(target_id)
        if target is None:
            continue
        if _dfgsb_visible_opcode(target.opcode):
            if all(existing.id != target.id for existing in visible):
                visible.append(target)
            continue
        if target_id in seen_hidden:
            continue
        seen_hidden.add(target_id)
        worklist.extend(outgoing.get(target_id, ()))
    return visible


def _bound_producer_node_ids(region: UHIRRegion) -> set[str]:
    producer_ids = set()
    producer_by_name = _producer_node_by_name(region)
    for binding in region.value_bindings:
        producer = producer_by_name.get(binding.producer)
        if producer is not None:
            producer_ids.add(producer.id)
    return producer_ids


def _local_register_display_id(producer: str, index: int, count: int) -> str:
    return f"reg_{producer}" if count == 1 else f"reg_{producer}_{index}"


def _port_name(text: str) -> str:
    return text.replace("@", "_at_").replace("-", "_").replace(":", "_")


def _hierarchy_region_order(design: UHIRDesign) -> list[str]:
    region_by_id = {region.id: region for region in design.regions}
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(region_id: str) -> None:
        if region_id in visited or region_id not in region_by_id:
            return
        visited.add(region_id)
        ordered.append(region_id)
        region = region_by_id[region_id]
        for node in region.nodes:
            for child_id in _node_children(node):
                visit(child_id)

    for region_id in sorted(region.id for region in design.regions if region.parent is None):
        visit(region_id)
    return ordered


def _region_boundary_line(region: UHIRRegion, region_by_id: dict[str, UHIRRegion]) -> str:
    return f"----- {_region_boundary_label(region, region_by_id)} -----"


def _region_boundary_label(region: UHIRRegion, region_by_id: dict[str, UHIRRegion]) -> str:
    parts = [region.id, f"kind={region.kind}"]
    owner = _owner_node(region, region_by_id)
    if owner is None:
        return " ".join(parts)
    parts.append(f"via={owner.opcode}:{owner.id}")
    role = _owner_child_role(owner, region.id)
    if role:
        parts.append(role)
    static_trip_count = owner.attributes.get("static_trip_count")
    if isinstance(static_trip_count, int):
        parts.append(f"iterations={static_trip_count}")
    return " ".join(parts)


def _owner_node(region: UHIRRegion, region_by_id: dict[str, UHIRRegion]) -> UHIRNode | None:
    if region.parent is None:
        return None
    parent = region_by_id.get(region.parent)
    if parent is None:
        return None
    for node in parent.nodes:
        if region.id in _node_children(node):
            return node
    return None


def _owner_child_role(owner: UHIRNode, region_id: str) -> str | None:
    for key in ("child", "true_child", "false_child"):
        child_id = owner.attributes.get(key)
        if child_id == region_id:
            return f"{key}={region_id}"
    return None


def _resource_color_map(resource_ids: list[str]) -> dict[str, str]:
    palette = (
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
    )
    ordered = sorted(set(resource_ids))
    return {resource_id: palette[index % len(palette)] for index, resource_id in enumerate(ordered)}


def _analysis_bind(node: UHIRNode, class_name: str) -> str:
    bind = node.attributes.get("bind")
    if isinstance(bind, str):
        return bind
    return class_name.lower()


def _validate_bind_design(design: UHIRDesign) -> None:
    if design.stage != "bind":
        raise ValueError(f"bind dump rendering expects bind-stage µhIR input, got stage '{design.stage}'")


def _escape_dot(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _escape_dot_label(text: str) -> str:
    return text.replace('"', '\\"')


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
