"""Analysis dump helpers for bind-stage operation binding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from uhls.backend.uhir.model import UHIRDesign, UHIRNode, UHIRRegion
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
    for scope_id, entries in sorted(_group_by_scope(_collect_local_bound_entries(design)).items()):
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
    grouped = _group_by_scope(_filter_dfgsb_entries(design, _collect_local_bound_entries(design)))
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
    for scope_id, entries in sorted(_group_by_scope(_collect_local_bound_entries(design)).items()):
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
    grouped = _group_by_scope(_filter_dfgsb_entries(design, _collect_local_bound_entries(design)))
    for region_id in _hierarchy_region_order(design):
        region = region_by_id[region_id]
        cluster_id = f"dfgsb_{region_id}"
        layout_lines, layout = _render_dfgsb_dot_scope(
            cluster_id=cluster_id,
            label=_region_boundary_label(region, region_by_id),
            entries=grouped.get(region_id, []),
            compact=compact,
        )
        layouts[region_id] = layout
        lines.extend(layout_lines)
    lines.extend(_dot_dfgsb_region_edges(design, layouts))
    return lines


def _dot_dfgsb_unroll(design: UHIRDesign, *, compact: bool) -> list[str]:
    entries = _filter_dfgsb_entries(design, _collect_scroll_bound_entries(design))
    lines, layout = _render_dfgsb_dot_scope(
        cluster_id="dfgsb_unroll",
        label="global dataflow graph with schedule and binding (unrolled)",
        entries=entries,
        compact=compact,
    )
    lines.extend(_dot_dfgsb_unrolled_edges(design, layout, entries))
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


def _collect_scroll_bound_entries(design: UHIRDesign) -> list[_BoundOccurrence]:
    helper = OperationBinderBase()
    region_by_id = {region.id: region for region in design.regions}
    entries: list[_BoundOccurrence] = []

    def visit_region(region_id: str, offset: int, *, suffix: str = "") -> None:
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
            entries.append(
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
                )
        for binding in region.value_bindings:
            register_id = binding.register
            register_resource = next((resource for resource in design.resources if resource.id == register_id), None)
            if register_resource is None or register_resource.kind != "reg":
                continue
            if binding.producer not in local_value_ids:
                continue
            for index, (start, end) in enumerate(binding.live_intervals):
                display_id = binding.producer if not suffix else f"{binding.producer}{suffix}"
                if len(binding.live_intervals) > 1:
                    display_id = f"{display_id}_{index}"
                entries.append(
                    _BoundOccurrence(
                        category="register",
                        scope_id="global",
                        region_id=region.id,
                        node_id=binding.producer,
                        display_id=f"reg_{display_id}",
                        opcode="reg",
                        bind=register_id,
                        class_name=register_resource.value,
                        start=start + offset,
                        end=end + offset,
                    )
                )

        for node in region.nodes:
            node_start = node.attributes.get("start")
            if not isinstance(node_start, int):
                continue
            if node.opcode == "loop":
                child_id = node.attributes.get("child")
                trip_count = node.attributes.get("static_trip_count")
                iter_ii = node.attributes.get("iter_initiation_interval")
                if not isinstance(child_id, str) or not isinstance(trip_count, int) or not isinstance(iter_ii, int):
                    raise ValueError(
                        f"loop node '{node.id}' requires child/static_trip_count/iter_initiation_interval for unrolled bind dumps"
                    )
                for iteration in range(trip_count):
                    visit_loop_header(child_id, offset + node_start + iteration * iter_ii, iteration, expand_body=True)
                visit_loop_header(child_id, offset + node_start + trip_count * iter_ii, trip_count, expand_body=False)
                continue

            for child_id in _node_children(node):
                # Child regions in sched/bind µhIR are already shifted into their
                # parent-local time base, so the scroll expansion only reapplies
                # the enclosing scope offset once.
                visit_region(child_id, offset, suffix=suffix)

    def visit_loop_header(region_id: str, offset: int, iteration: int, *, expand_body: bool) -> None:
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
            entries.append(
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
                display_id = f"{binding.producer}{suffix}" if len(binding.live_intervals) == 1 else f"{binding.producer}{suffix}_{index}"
                entries.append(
                    _BoundOccurrence(
                        category="register",
                        scope_id="global",
                        region_id=region.id,
                        node_id=binding.producer,
                        display_id=f"reg_{display_id}",
                        opcode="reg",
                        bind=register_id,
                        class_name=register_resource.value,
                        start=start + offset,
                        end=end + offset,
                    )
                )
        for node in region.nodes:
            node_start = node.attributes.get("start")
            if not isinstance(node_start, int):
                continue
            if node.opcode == "branch":
                true_child = node.attributes.get("true_child")
                if expand_body and isinstance(true_child, str):
                    visit_region(true_child, offset, suffix=suffix)
                # TODO: Add explicit expansion for loop exit regions once they may contain bindable work.
                continue
            for child_id in _node_children(node):
                visit_region(child_id, offset, suffix=suffix)

    for region_id in sorted(region.id for region in design.regions if region.parent is None):
        visit_region(region_id, 0)
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
    compact: bool,
) -> tuple[list[str], _DFGSBDotLayout]:
    lines = [
        f'  subgraph "cluster_{cluster_id}" {{',
        f'    label="{_escape_dot(label)}";',
        "    color=gray70;",
    ]
    layout = _DFGSBDotLayout(cluster_id=cluster_id, op_nodes={}, reg_nodes={})
    if not entries:
        lines.append(f'    "{cluster_id}_empty" [label="empty", shape=plaintext, style=""];')
        lines.append("  }")
        return lines, layout

    lanes = _dfgsb_lane_order(entries)
    resource_colors = _resource_color_map(lanes)
    lane_headers = []
    for index, lane in enumerate(lanes):
        header_id = f"{cluster_id}_lane_{index}"
        lane_headers.append(header_id)
        lines.append(f'    "{header_id}" [label="{_escape_dot(lane)}", shape=plaintext, style=""];')
    lines.append("    { rank=same; " + " ".join(f'"{node_id}";' for node_id in lane_headers) + " }")
    for left_id, right_id in zip(lane_headers, lane_headers[1:]):
        lines.append(f'    "{left_id}" -> "{right_id}" [style=invis, weight=50];')

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
            label = f"{entry.display_id}\\n{opcode}\\n{entry.bind}"
            shape = "box"
            layout.op_nodes[(entry.region_id, entry.display_id)] = node_id
        else:
            label = f"{entry.node_id}\\n{entry.bind}"
            shape = "box"
            layout.reg_nodes[(entry.region_id, entry.display_id, time_step)] = node_id
        return node_id, [f'    "{node_id}" [label="{_escape_dot_label(label)}", shape={shape}, fillcolor="{fillcolor}"];']

    joined = "\\n".join(_entry_body(entry, compact=compact) for entry in occupants)
    return node_id, [f'    "{node_id}" [label="{_escape_dot_label(joined)}", shape=box, fillcolor="{fillcolor}"];']


def _dot_dfgsb_region_edges(design: UHIRDesign, layouts: dict[str, _DFGSBDotLayout]) -> list[str]:
    helper = OperationBinderBase()
    lines: list[str] = []
    for region in design.regions:
        layout = layouts.get(region.id)
        if layout is None:
            continue
        lines.extend(_dot_dfgsb_region_data_edges(region, layout, helper))
        lines.extend(_dot_dfgsb_region_register_edges(design, region, layout, layouts, helper))
    return lines


def _dot_dfgsb_region_data_edges(region: UHIRRegion, layout: _DFGSBDotLayout, helper: OperationBinderBase) -> list[str]:
    lines: list[str] = []
    bound_producers = _bound_producer_node_ids(region)
    for edge in helper.iter_region_data_edges(region):
        if edge.source in bound_producers:
            continue
        source_id = layout.op_nodes.get((region.id, edge.source))
        target_id = layout.op_nodes.get((region.id, edge.target))
        if source_id is None or target_id is None:
            continue
        lines.append(f'  "{source_id}" -> "{target_id}" [color="#444444", penwidth=1.3];')
    return lines


def _dot_dfgsb_region_register_edges(
    design: UHIRDesign,
    region: UHIRRegion,
    layout: _DFGSBDotLayout,
    layouts: dict[str, _DFGSBDotLayout],
    helper: OperationBinderBase,
) -> list[str]:
    lines: list[str] = []
    producer_by_name = _producer_node_by_name(region)
    for binding in region.value_bindings:
        producer = producer_by_name.get(binding.producer)
        if producer is None or not _dfgsb_visible_opcode(producer.opcode):
            continue
        producer_id = layout.op_nodes.get((region.id, producer.id))
        if producer_id is None:
            continue
        consumers = [consumer for consumer in _dfgsb_value_consumers(design, region, producer, helper) if _dfgsb_visible_opcode(consumer.node.opcode)]
        for index, (start, end) in enumerate(binding.live_intervals):
            reg_display = _local_register_display_id(binding.producer, index, len(binding.live_intervals))
            first_reg_id = layout.reg_nodes.get((region.id, reg_display, start))
            if first_reg_id is not None:
                lines.append(f'  "{producer_id}" -> "{first_reg_id}" [color="#1f78b4", penwidth=1.3];')
            for step in range(start, end):
                left_id = layout.reg_nodes.get((region.id, reg_display, step))
                right_id = layout.reg_nodes.get((region.id, reg_display, step + 1))
                if left_id is None or right_id is None:
                    continue
                lines.append(f'  "{left_id}" -> "{right_id}" [color="#1f78b4", penwidth=1.1];')
            for consumer in consumers:
                target_layout = layouts.get(consumer.region.id)
                if target_layout is None:
                    continue
                target_row = helper.get_node_interval(consumer.region, consumer.node)[0] + consumer.start_shift
                consumer_id = target_layout.op_nodes.get((consumer.region.id, consumer.node.id))
                if consumer_id is None:
                    continue
                if target_row < start:
                    lines.append(f'  "{producer_id}" -> "{consumer_id}" [color="#1f78b4", penwidth=1.3];')
                    continue
                reg_id = layout.reg_nodes.get((region.id, reg_display, target_row))
                if reg_id is None:
                    continue
                lines.append(f'  "{reg_id}" -> "{consumer_id}" [color="#1f78b4", penwidth=1.3];')
    return lines


def _dot_dfgsb_unrolled_edges(design: UHIRDesign, layout: _DFGSBDotLayout, entries: list[_BoundOccurrence]) -> list[str]:
    helper = OperationBinderBase()
    lines: list[str] = []
    op_index: dict[tuple[str, str, str], str] = {}
    for entry in entries:
        if entry.category != "operation":
            continue
        op_node_id = layout.op_nodes.get((entry.region_id, entry.display_id))
        if op_node_id is None:
            continue
        op_index[(entry.region_id, entry.node_id, _display_suffix(entry.node_id, entry.display_id))] = op_node_id

    for region in design.regions:
        bound_producers = _bound_producer_node_ids(region)
        for edge in helper.iter_region_data_edges(region):
            if edge.source in bound_producers:
                continue
            if not (_dfgsb_visible_node_id(entries, region.id, edge.source) and _dfgsb_visible_node_id(entries, region.id, edge.target)):
                continue
            for suffix in _suffixes_for_region_node(entries, region.id, edge.source):
                source_id = op_index.get((region.id, edge.source, suffix))
                target_id = op_index.get((region.id, edge.target, suffix))
                if source_id is None or target_id is None:
                    continue
                lines.append(f'  "{source_id}" -> "{target_id}" [color="#444444", penwidth=1.3];')

        producer_by_name = _producer_node_by_name(region)
        for binding in region.value_bindings:
            producer = producer_by_name.get(binding.producer)
            if producer is None or not _dfgsb_visible_opcode(producer.opcode):
                continue
            consumers = [consumer for consumer in _dfgsb_value_consumers(design, region, producer, helper) if _dfgsb_visible_opcode(consumer.node.opcode)]
            producer_base_start = helper.get_node_interval(region, producer)[0]
            for suffix in _suffixes_for_region_node(entries, region.id, producer.id):
                producer_entry = next(
                    (
                        entry
                        for entry in entries
                        if entry.category == "operation"
                        and entry.region_id == region.id
                        and entry.node_id == producer.id
                        and _display_suffix(producer.id, entry.display_id) == suffix
                    ),
                    None,
                )
                if producer_entry is None:
                    continue
                producer_id = layout.op_nodes.get((producer_entry.region_id, producer_entry.display_id))
                if producer_id is None:
                    continue
                occurrence_offset = producer_entry.start - producer_base_start
                for index, (start, end) in enumerate(binding.live_intervals):
                    reg_display = _suffixed_register_display_id(binding.producer, suffix, index, len(binding.live_intervals))
                    shifted_start = start + occurrence_offset
                    shifted_end = end + occurrence_offset
                    first_reg_id = _find_unrolled_register_node(layout, region.id, reg_display, shifted_start)
                    if first_reg_id is not None:
                        lines.append(f'  "{producer_id}" -> "{first_reg_id}" [color="#1f78b4", penwidth=1.3];')
                    for step in range(shifted_start, shifted_end):
                        left_id = _find_unrolled_register_node(layout, region.id, reg_display, step)
                        right_id = _find_unrolled_register_node(layout, region.id, reg_display, step + 1)
                        if left_id is None or right_id is None:
                            continue
                        lines.append(f'  "{left_id}" -> "{right_id}" [color="#1f78b4", penwidth=1.1];')
                    for consumer in consumers:
                        target_time = _dfgsb_unrolled_consumer_time(
                            design,
                            region,
                            consumer,
                            helper,
                            occurrence_offset,
                        )
                        target_entry = _find_unrolled_consumer_entry(entries, consumer.region.id, consumer.node.id, target_time)
                        if target_entry is None:
                            continue
                        consumer_id = layout.op_nodes.get((target_entry.region_id, target_entry.display_id))
                        if consumer_id is None:
                            continue
                        if target_time < shifted_start:
                            lines.append(f'  "{producer_id}" -> "{consumer_id}" [color="#1f78b4", penwidth=1.3];')
                            continue
                        reg_id = _find_unrolled_register_node(layout, region.id, reg_display, target_time)
                        if reg_id is None:
                            continue
                        lines.append(f'  "{reg_id}" -> "{consumer_id}" [color="#1f78b4", penwidth=1.3];')
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
    return f"{entry.display_id} {opcode} {entry.bind}"


def _entry_body(entry: _BoundOccurrence, *, compact: bool) -> str:
    if entry.category == "register":
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


def _dfgsb_lane_order(entries: list[_BoundOccurrence]) -> list[str]:
    operation_lanes = sorted({entry.bind for entry in entries if entry.category == "operation"})
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
            if entry.category == "operation" and entry.bind == lane and entry.start <= time_step <= entry.end
        ]
    return [
        entry
        for entry in entries
        if entry.category == "register" and entry.bind == lane and entry.start <= time_step <= entry.end
    ]


def _producer_node_by_name(region: UHIRRegion) -> dict[str, UHIRNode]:
    producer_by_name = {node.id: node for node in region.nodes}
    for mapping in region.mappings:
        node = next((candidate for candidate in region.nodes if candidate.id == mapping.node_id), None)
        if node is not None:
            producer_by_name[mapping.source_id] = node
    return producer_by_name


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


def _dfgsb_unrolled_consumer_time(
    design: UHIRDesign,
    producer_region: UHIRRegion,
    consumer: ValueConsumerRef,
    helper: OperationBinderBase,
    occurrence_offset: int,
) -> int:
    base_time = helper.get_node_interval(consumer.region, consumer.node)[0] + consumer.start_shift
    if (
        consumer.region.id != producer_region.id
        and _is_ancestor_region(design, consumer.region.id, producer_region.id)
        and consumer.node.opcode != "phi"
    ):
        return base_time
    return base_time + occurrence_offset


def _is_ancestor_region(design: UHIRDesign, ancestor_id: str, region_id: str) -> bool:
    current_id = region_id
    while True:
        region = design.get_region(current_id)
        if region is None or region.parent is None:
            return False
        if region.parent == ancestor_id:
            return True
        current_id = region.parent


def _bound_producer_node_ids(region: UHIRRegion) -> set[str]:
    producer_ids = set()
    producer_by_name = _producer_node_by_name(region)
    for binding in region.value_bindings:
        producer = producer_by_name.get(binding.producer)
        if producer is not None:
            producer_ids.add(producer.id)
    return producer_ids


def _dfgsb_visible_node_id(entries: list[_BoundOccurrence], region_id: str, node_id: str) -> bool:
    return any(
        entry.category == "operation" and entry.region_id == region_id and entry.node_id == node_id
        for entry in entries
    )


def _display_suffix(node_id: str, display_id: str) -> str:
    if display_id == node_id:
        return ""
    if display_id.startswith(node_id):
        return display_id[len(node_id) :]
    return ""


def _suffixes_for_region_node(entries: list[_BoundOccurrence], region_id: str, node_id: str) -> list[str]:
    suffixes = {
        _display_suffix(node_id, entry.display_id)
        for entry in entries
        if entry.category == "operation" and entry.region_id == region_id and entry.node_id == node_id
    }
    return sorted(suffixes)


def _local_register_display_id(producer: str, index: int, count: int) -> str:
    return f"reg_{producer}" if count == 1 else f"reg_{producer}_{index}"


def _suffixed_register_display_id(producer: str, suffix: str, index: int, count: int) -> str:
    base = f"reg_{producer}{suffix}"
    return base if count == 1 else f"{base}_{index}"


def _find_unrolled_consumer_entry(
    entries: list[_BoundOccurrence],
    region_id: str,
    node_id: str,
    target_time: int,
) -> _BoundOccurrence | None:
    candidates = [
        entry
        for entry in entries
        if entry.category == "operation" and entry.region_id == region_id and entry.node_id == node_id and entry.start == target_time
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda entry: entry.display_id)
    return candidates[0]


def _find_unrolled_register_node(
    layout: _DFGSBDotLayout,
    region_id: str,
    reg_display: str,
    time_step: int,
) -> str | None:
    return layout.reg_nodes.get((region_id, reg_display, time_step))


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
