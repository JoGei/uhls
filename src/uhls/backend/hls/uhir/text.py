"""Text parsing helpers for textual µhIR artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from uhls.middleend.uir import COMPACT_OPCODE_LABELS

from .model import (
    AttributeValue,
    UHIRConstant,
    UHIRController,
    UHIRControllerEmit,
    UHIRControllerLink,
    UHIRControllerState,
    UHIRControllerTransition,
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
from .timing import TimingExpr, parse_timing_expr

_IDENT_RE = r"[A-Za-z_][\w$]*"
_RESOURCE_VALUE_RE = r"[A-Za-z_][\w$]*(?:<[^>\n]+>)?"
_DESIGN_RE = re.compile(rf"^design\s+({_IDENT_RE})$")
_STAGE_RE = re.compile(r"^stage\s+(seq|exg|alloc|sched|bind|fsm)$")
_COMPONENT_LIBRARY_RE = re.compile(r"^component_library\s+(.+)$")
_PORT_RE = re.compile(rf"^(input|output)\s+({_IDENT_RE})\s*:\s*(.+)$")
_CONTROLLER_START_RE = re.compile(rf"^controller\s+({_IDENT_RE})(?:\s+(.*?))?\{{$")
_STATE_RE = re.compile(rf"^state\s+({_IDENT_RE})(?:\s+(.*))?$")
_TRANSITION_RE = re.compile(rf"^transition\s+({_IDENT_RE})\s*->\s*({_IDENT_RE})(?:\s+(.*))?$")
_EMIT_RE = re.compile(rf"^emit\s+({_IDENT_RE})(?:\s+(.*))?$")
_LINK_RE = re.compile(rf"^link\s+({_IDENT_RE})(?:\s+(.*))?$")
_CONST_RE = re.compile(rf"^const\s+({_IDENT_RE})\s*=\s*(.+?)\s*:\s*(.+)$")
_SCHEDULE_RE = re.compile(rf"^schedule\s+kind=({_IDENT_RE})$")
_REGION_START_RE = re.compile(rf"^region\s+({_IDENT_RE})\s+(.+)\{{$")
_REGION_REF_RE = re.compile(rf"^region_ref\s+({_IDENT_RE})$")
_NODE_RE = re.compile(rf"^node\s+({_IDENT_RE})\s*=\s*(.+)$")
_EDGE_RE = re.compile(rf"^edge\s+({_IDENT_RE})\s+(\S+)\s*(->|--)\s*(\S+)(?:\s+(.*))?$")
_MAP_RE = re.compile(rf"^map\s+({_IDENT_RE})\s*<-\s*(\S+)$")
_STEPS_RE = re.compile(r"^steps\s+(.+)$")
_LATENCY_RE = re.compile(r"^latency\s+(.+)$")
_II_RE = re.compile(r"^ii\s+(.+)$")
_VALUE_RE = re.compile(r"^value\s+(\S+)\s*->\s*(\S+)\s+(.+)$")
_MUX_RE = re.compile(rf"^mux\s+({_IDENT_RE})\s*:\s*(.+)$")
_FU_RE = re.compile(rf"^fu\s+({_IDENT_RE})\s*:\s*({_RESOURCE_VALUE_RE})$")
_REG_RE = re.compile(rf"^reg\s+({_IDENT_RE})\s*:\s*([A-Za-z0-9_<>\[\]]+)$")
_NET_RE = re.compile(rf"^net\s+({_IDENT_RE})\s*:\s*([A-Za-z0-9_<>\[\]]+)$")
_MEM_RE = re.compile(rf"^mem\s+({_IDENT_RE})\s*:\s*([A-Za-z0-9_<>\[\]]+)$")
_INST_RE = re.compile(rf"^inst\s+({_IDENT_RE})\s*:\s*({_RESOURCE_VALUE_RE})$")
_MUX_RESOURCE_RE = re.compile(rf"^mux\s+({_IDENT_RE})\s*:\s*([A-Za-z0-9_<>\[\]]+)$")
_PORT_RESOURCE_RE = re.compile(rf"^port\s+({_IDENT_RE})\s*:\s*({_RESOURCE_VALUE_RE})(?:\s+({_IDENT_RE}))?$")
_UIR_LANGUAGE_OPCODES = frozenset(str(opcode).lower() for opcode in COMPACT_OPCODE_LABELS)


class UHIRParseError(ValueError):
    """Raised when textual µhIR cannot be parsed."""


def parse_uhir(text: str) -> UHIRDesign:
    """Parse one textual µhIR artifact."""
    lines = _normalize_lines(text)
    if not lines:
        raise UHIRParseError("empty µhIR input")

    index = 0
    name = _parse_design(lines[index], 1)
    index += 1
    if index >= len(lines):
        raise UHIRParseError("missing stage declaration")
    stage = _parse_stage(lines[index], index + 1)
    index += 1

    design = UHIRDesign(name=name, stage=stage)

    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        if component_library := _try_parse_component_library(line):
            design.component_libraries.append(component_library)
            index += 1
            continue
        if port := _try_parse_port(line):
            if port.direction == "input":
                design.inputs.append(port)
            else:
                design.outputs.append(port)
            index += 1
            continue
        if const_decl := _try_parse_const(line):
            design.constants.append(const_decl)
            index += 1
            continue
        if schedule := _try_parse_schedule(line):
            if design.schedule is not None:
                raise UHIRParseError(f"duplicate schedule declaration at line {line_number}")
            design.schedule = schedule
            index += 1
            continue
        if line.startswith("controller "):
            controller, index = _parse_controller(lines, index)
            design.controllers.append(controller)
            continue
        if line == "resources {":
            if design.resources:
                raise UHIRParseError(f"duplicate resources block at line {line_number}")
            resources, index = _parse_resources_block(lines, index)
            design.resources.extend(resources)
            continue
        if line.startswith("region "):
            region, index = _parse_region(lines, index)
            design.regions.append(region)
            continue
        raise UHIRParseError(f"unsupported top-level µhIR syntax at line {line_number}: {line!r}")

    _validate_design(design)
    return design


def parse_uhir_file(path: str | Path) -> UHIRDesign:
    """Parse one textual µhIR file from disk."""
    return parse_uhir(Path(path).read_text(encoding="utf-8"))


def _normalize_lines(text: str) -> list[str]:
    lines: list[str] = []
    buffer: list[str] = []
    paren_depth = 0
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if not buffer:
            buffer.append(line)
        else:
            buffer.append(line)
        paren_depth += _paren_delta(line)
        if paren_depth <= 0:
            lines.append(" ".join(buffer))
            buffer = []
            paren_depth = 0
    if buffer:
        lines.append(" ".join(buffer))
    return lines


def _strip_comment(text: str) -> str:
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if char == "#" and not in_string:
            return text[:index]
    return text


def _paren_delta(text: str) -> int:
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
    return depth


def _unwrap_parenthesized_expr(expr: str) -> str:
    text = expr.strip()
    while _is_fully_parenthesized(text):
        text = text[1:-1].strip()
    return text


def _is_fully_parenthesized(text: str) -> bool:
    if len(text) < 2 or text[0] != "(" or text[-1] != ")":
        return False
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
    return depth == 0


def _parse_design(line: str, line_number: int) -> str:
    match = _DESIGN_RE.fullmatch(line)
    if match is None:
        raise UHIRParseError(f"expected design declaration at line {line_number}: {line!r}")
    return match.group(1)


def _parse_stage(line: str, line_number: int) -> str:
    match = _STAGE_RE.fullmatch(line)
    if match is None:
        raise UHIRParseError(f"expected stage declaration at line {line_number}: {line!r}")
    return match.group(1)


def _try_parse_port(line: str) -> UHIRPort | None:
    match = _PORT_RE.fullmatch(line)
    if match is None:
        return None
    direction, name, type_hint = match.groups()
    return UHIRPort(direction, name, type_hint.strip())


def _try_parse_component_library(line: str) -> str | None:
    match = _COMPONENT_LIBRARY_RE.fullmatch(line)
    if match is None:
        return None
    value = match.group(1).strip()
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise UHIRParseError(f"component_library expects one valid string path, got {value!r}") from exc
        if not isinstance(parsed, str):
            raise UHIRParseError(f"component_library expects one string path, got {value!r}")
        return parsed
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return value[1:-1]
    return value


def _try_parse_const(line: str) -> UHIRConstant | None:
    match = _CONST_RE.fullmatch(line)
    if match is None:
        return None
    name, value_text, type_hint = match.groups()
    return UHIRConstant(name, _parse_scalar_text(_unwrap_parenthesized_expr(value_text.strip())), type_hint.strip())


def _try_parse_schedule(line: str) -> UHIRSchedule | None:
    match = _SCHEDULE_RE.fullmatch(line)
    if match is None:
        return None
    return UHIRSchedule(kind=match.group(1))


def _parse_resources_block(lines: list[str], index: int) -> tuple[list[UHIRResource], int]:
    resources: list[UHIRResource] = []
    index += 1
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        if line == "}":
            return resources, index + 1
        if match := _FU_RE.fullmatch(line):
            resources.append(UHIRResource("fu", match.group(1), match.group(2)))
        elif match := _REG_RE.fullmatch(line):
            resources.append(UHIRResource("reg", match.group(1), match.group(2)))
        elif match := _NET_RE.fullmatch(line):
            resources.append(UHIRResource("net", match.group(1), match.group(2)))
        elif match := _MEM_RE.fullmatch(line):
            resources.append(UHIRResource("mem", match.group(1), match.group(2)))
        elif match := _INST_RE.fullmatch(line):
            resources.append(UHIRResource("inst", match.group(1), match.group(2)))
        elif match := _MUX_RESOURCE_RE.fullmatch(line):
            resources.append(UHIRResource("mux", match.group(1), match.group(2)))
        elif match := _PORT_RESOURCE_RE.fullmatch(line):
            resources.append(UHIRResource("port", match.group(1), match.group(2), match.group(3)))
        else:
            raise UHIRParseError(f"invalid resources declaration at line {line_number}: {line!r}")
        index += 1
    raise UHIRParseError("unterminated resources block")


def _parse_region(lines: list[str], index: int) -> tuple[UHIRRegion, int]:
    line = lines[index]
    line_number = index + 1
    match = _REGION_START_RE.fullmatch(line)
    if match is None:
        raise UHIRParseError(f"invalid region declaration at line {line_number}: {line!r}")
    region_id, attrs_text = match.groups()
    attrs = _parse_attrs(attrs_text.strip())
    kind = attrs.pop("kind", None)
    if not isinstance(kind, str) or not kind:
        raise UHIRParseError(f"region '{region_id}' is missing kind=... at line {line_number}")
    parent = attrs.pop("parent", None)
    if attrs:
        unknown = ", ".join(sorted(attrs))
        raise UHIRParseError(f"unsupported region attributes at line {line_number}: {unknown}")

    region = UHIRRegion(id=region_id, kind=kind, parent=parent if isinstance(parent, str) else None)
    index += 1
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        if line == "}":
            return region, index + 1
        if match := _REGION_REF_RE.fullmatch(line):
            region.region_refs.append(UHIRRegionRef(match.group(1)))
        elif match := _NODE_RE.fullmatch(line):
            region.nodes.append(_parse_node(match.group(1), match.group(2), line_number))
        elif match := _EDGE_RE.fullmatch(line):
            kind_text, source, edge_op, target, attrs_text = match.groups()
            region.edges.append(
                UHIREdge(
                    kind=kind_text,
                    source=source,
                    target=target,
                    attributes=_parse_attrs(attrs_text or ""),
                    directed=edge_op == "->",
                )
            )
        elif match := _MAP_RE.fullmatch(line):
            region.mappings.append(UHIRSourceMap(match.group(1), match.group(2)))
        elif match := _STEPS_RE.fullmatch(line):
            region.steps = _parse_steps_interval(match.group(1), line_number)
        elif match := _LATENCY_RE.fullmatch(line):
            region.latency = _parse_timing_scalar_text(match.group(1), line_number, "latency")
        elif match := _II_RE.fullmatch(line):
            region.initiation_interval = _parse_timing_scalar_text(match.group(1), line_number, "ii")
        elif match := _VALUE_RE.fullmatch(line):
            region.value_bindings.append(_parse_value_binding(match.group(1), match.group(2), match.group(3), line_number))
        elif match := _MUX_RE.fullmatch(line):
            region.muxes.append(_parse_mux(match.group(1), match.group(2), line_number))
        else:
            raise UHIRParseError(f"unsupported region-local µhIR syntax at line {line_number}: {line!r}")
        index += 1
    raise UHIRParseError(f"unterminated region '{region_id}'")


def _parse_controller(lines: list[str], index: int) -> tuple[UHIRController, int]:
    line = lines[index]
    line_number = index + 1
    match = _CONTROLLER_START_RE.fullmatch(line)
    if match is None:
        raise UHIRParseError(f"invalid controller declaration at line {line_number}: {line!r}")
    controller_name, attrs_text = match.groups()
    controller = UHIRController(
        name=controller_name,
        attributes=_parse_attrs("" if attrs_text is None else attrs_text.strip()),
    )
    index += 1
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        if line == "}":
            return controller, index + 1
        if port := _try_parse_port(line):
            if port.direction == "input":
                controller.inputs.append(port)
            else:
                controller.outputs.append(port)
            index += 1
            continue
        if match := _STATE_RE.fullmatch(line):
            controller.states.append(
                UHIRControllerState(
                    name=match.group(1),
                    attributes=_parse_attrs("" if match.group(2) is None else match.group(2).strip()),
                )
            )
            index += 1
            continue
        if match := _TRANSITION_RE.fullmatch(line):
            controller.transitions.append(
                UHIRControllerTransition(
                    source=match.group(1),
                    target=match.group(2),
                    attributes=_parse_attrs("" if match.group(3) is None else match.group(3).strip()),
                )
            )
            index += 1
            continue
        if match := _EMIT_RE.fullmatch(line):
            controller.emits.append(
                UHIRControllerEmit(
                    state=match.group(1),
                    attributes=_parse_attrs("" if match.group(2) is None else match.group(2).strip()),
                )
            )
            index += 1
            continue
        if match := _LINK_RE.fullmatch(line):
            attrs = _parse_attrs("" if match.group(2) is None else match.group(2).strip())
            node = attrs.pop("via", None)
            if not isinstance(node, str) or not node:
                raise UHIRParseError(f"controller link '{match.group(1)}' is missing via=<node-id> at line {line_number}")
            controller.links.append(
                UHIRControllerLink(
                    child=match.group(1),
                    node=node,
                    attributes=attrs,
                )
            )
            index += 1
            continue
        raise UHIRParseError(f"unsupported controller-local µhIR syntax at line {line_number}: {line!r}")
    raise UHIRParseError(f"unterminated controller '{controller_name}'")


def _parse_node(node_id: str, remainder: str, line_number: int) -> UHIRNode:
    body, attrs = _split_attr_suffix(remainder)
    expr_text, result_type = _split_node_type(body)
    if not expr_text:
        raise UHIRParseError(f"node '{node_id}' is missing an expression at line {line_number}")
    parts = expr_text.split(None, 1)
    opcode = _normalize_node_opcode(parts[0])
    operands = () if len(parts) == 1 else tuple(_split_top_level(parts[1]))
    return UHIRNode(node_id, opcode, operands, result_type, attrs)


def _normalize_node_opcode(opcode: str) -> str:
    if opcode == "select":
        return "sel"
    return opcode


def _parse_value_binding(producer: str, register: str, attrs_text: str, line_number: int) -> UHIRValueBinding:
    attrs = _parse_attrs(attrs_text)
    live = attrs.pop("live", None)
    if attrs:
        unknown = ", ".join(sorted(attrs))
        raise UHIRParseError(f"unsupported value binding attributes at line {line_number}: {unknown}")
    if isinstance(live, tuple):
        normalized_live = ",".join(f"[{str(item).strip()}]" for item in live)
        return UHIRValueBinding(producer, register, _parse_live_intervals(normalized_live, line_number))
    if not isinstance(live, str):
        raise UHIRParseError(f"value binding is missing live=... at line {line_number}")
    return UHIRValueBinding(producer, register, _parse_live_intervals(live, line_number))


def _parse_steps_interval(text: str, line_number: int) -> tuple[int | str, int | str]:
    normalized = text.strip()
    if not (normalized.startswith("[") and normalized.endswith("]")):
        raise UHIRParseError(f"steps interval must use [start:end] syntax at line {line_number}: {normalized!r}")
    body = normalized[1:-1].strip()
    if ":" not in body:
        raise UHIRParseError(f"steps interval must use [start:end] syntax at line {line_number}: {normalized!r}")
    start_text, end_text = (piece.strip() for piece in body.split(":", 1))
    start = _parse_timing_scalar_text(start_text, line_number, "steps start")
    end = _parse_timing_scalar_text(end_text, line_number, "steps end")
    if isinstance(start, int) and isinstance(end, int) and start > end:
        raise UHIRParseError(f"steps interval must satisfy start<=end at line {line_number}: {normalized!r}")
    return start, end


def _parse_timing_scalar_text(text: str, line_number: int, context: str) -> int | str:
    normalized = text.strip()
    if not normalized:
        raise UHIRParseError(f"{context} must not be empty at line {line_number}")
    try:
        return parse_timing_expr(normalized)
    except ValueError as exc:
        raise UHIRParseError(f"invalid {context} timing expression at line {line_number}: {normalized!r}") from exc


def _parse_live_intervals(text: str, line_number: int) -> tuple[tuple[int, int], ...]:
    normalized = text.strip()
    intervals: list[tuple[int, int]] = []
    for item in _split_top_level(normalized):
        part = item.strip()
        if not part.startswith("[") or not part.endswith("]"):
            raise UHIRParseError(f"value binding live interval must use [start:end] syntax at line {line_number}: {part!r}")
        body = part[1:-1].strip()
        if ":" not in body:
            raise UHIRParseError(f"value binding live interval is missing ':' at line {line_number}: {part!r}")
        start_text, end_text = (piece.strip() for piece in body.split(":", 1))
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError as exc:
            raise UHIRParseError(f"value binding live interval must use integer bounds at line {line_number}: {part!r}") from exc
        if start > end:
            raise UHIRParseError(f"value binding interval must satisfy start<=end at line {line_number}: {part!r}")
        intervals.append((start, end))
    if not intervals:
        raise UHIRParseError(f"value binding is missing one or more live intervals at line {line_number}")
    return tuple(intervals)


def _parse_mux(mux_id: str, attrs_text: str, line_number: int) -> UHIRMux:
    attrs = _parse_attrs(attrs_text)
    inputs = attrs.pop("input", None)
    output = attrs.pop("output", None)
    select = attrs.pop("sel", None)
    if not isinstance(inputs, tuple) or not all(isinstance(item, str) for item in inputs):
        raise UHIRParseError(f"mux '{mux_id}' is missing input=[...] at line {line_number}")
    if not isinstance(output, str) or not output:
        raise UHIRParseError(f"mux '{mux_id}' is missing output=... at line {line_number}")
    if not isinstance(select, str) or not select:
        raise UHIRParseError(f"mux '{mux_id}' is missing sel=... at line {line_number}")
    return UHIRMux(mux_id, inputs, output, select, attrs)


def _split_node_type(text: str) -> tuple[str, str | None]:
    depth = 0
    split_index: int | None = None
    for index, char in enumerate(text):
        if _is_nesting_open(text, index):
            depth += 1
        elif _is_nesting_close(text, index):
            depth -= 1
        elif char == ":" and depth == 0:
            split_index = index
    if split_index is None:
        return text.strip(), None
    return text[:split_index].strip(), text[split_index + 1 :].strip() or None


def _split_attr_suffix(text: str) -> tuple[str, dict[str, AttributeValue]]:
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if _is_nesting_open(text, index):
            depth += 1
            continue
        if _is_nesting_close(text, index):
            depth -= 1
            continue
        if char.isspace() and depth == 0:
            candidate_index = _skip_spaces(text, index)
            if candidate_index < len(text) and _looks_like_attr_start(text, candidate_index):
                return text[:index].strip(), _parse_attrs(text[candidate_index:])
    return text.strip(), {}


def _parse_attrs(text: str) -> dict[str, AttributeValue]:
    attrs: dict[str, AttributeValue] = {}
    index = 0
    while index < len(text):
        index = _skip_spaces(text, index)
        if index >= len(text):
            break
        key_start = index
        while index < len(text) and text[index] not in "= ":
            index += 1
        key = text[key_start:index]
        if not key or index >= len(text) or text[index] != "=":
            raise UHIRParseError(f"invalid attribute spelling {text!r}")
        index += 1
        value_start = index
        depth = 0
        in_string = False
        escaped = False
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if char == "\\" and in_string:
                escaped = True
                index += 1
                continue
            if char == '"':
                in_string = not in_string
                index += 1
                continue
            if in_string:
                index += 1
                continue
            if _is_nesting_open(text, index):
                depth += 1
            elif _is_nesting_close(text, index):
                depth -= 1
            elif char.isspace() and depth == 0:
                next_index = _skip_spaces(text, index)
                if next_index >= len(text) or _looks_like_attr_start(text, next_index):
                    break
            index += 1
        value_text = text[value_start:index].strip()
        attrs[key] = _parse_attr_value(value_text)
    return attrs


def _parse_attr_value(text: str) -> AttributeValue:
    if text.startswith("[") and text.endswith("]"):
        inside = text[1:-1].strip()
        return tuple(_split_top_level(inside)) if inside else ()
    if text == "true":
        return True
    if text == "false":
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def _parse_scalar_text(text: str) -> int | str:
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def _looks_like_attr_start(text: str, index: int) -> bool:
    if not (text[index].isalpha() or text[index] == "_"):
        return False
    cursor = index + 1
    while cursor < len(text) and (text[cursor].isalnum() or text[cursor] in "_$"):
        cursor += 1
    return cursor < len(text) and text[cursor] == "="


def _skip_spaces(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _split_top_level(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    items: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if _is_nesting_open(text, index):
            depth += 1
        elif _is_nesting_close(text, index):
            depth -= 1
        elif char == "," and depth == 0:
            items.append(text[start:index].strip())
            start = index + 1
    items.append(text[start:].strip())
    return items


def _is_nesting_open(text: str, index: int) -> bool:
    char = text[index]
    if char in "([":
        return True
    if char == "<":
        return index + 1 >= len(text) or text[index + 1] != "-"
    return False


def _is_nesting_close(text: str, index: int) -> bool:
    char = text[index]
    if char in ")]":
        return True
    if char == ">":
        return index == 0 or text[index - 1] != "-"
    return False


def _split_range(text: str, line_number: int) -> tuple[str, str]:
    if ".." not in text:
        raise UHIRParseError(f"invalid range syntax at line {line_number}: {text!r}")
    start_text, end_text = text.split("..", 1)
    return start_text.strip(), end_text.strip()


def _validate_design(design: UHIRDesign) -> None:
    if not design.regions:
        raise UHIRParseError("µhIR design must contain at least one region")
    if design.stage == "seq":
        if design.schedule is not None:
            raise UHIRParseError("seq µhIR must not declare a schedule block")
        if design.resources:
            raise UHIRParseError("seq µhIR must not declare resources")
    elif design.stage == "exg":
        if design.inputs or design.outputs or design.constants:
            raise UHIRParseError("exg µhIR must not declare ports or constants")
        if design.schedule is not None:
            raise UHIRParseError("exg µhIR must not declare a schedule block")
        if design.resources:
            raise UHIRParseError("exg µhIR must not declare resources")
    elif design.stage == "alloc":
        if design.schedule is not None:
            raise UHIRParseError("alloc µhIR must not declare a schedule block")
        if design.resources:
            raise UHIRParseError("alloc µhIR must not declare resources")
    elif design.stage == "sched":
        if design.schedule is None:
            raise UHIRParseError("sched µhIR must declare schedule kind=...")
        if design.resources:
            raise UHIRParseError("sched µhIR must not declare resources")
    elif design.stage == "bind":
        if design.schedule is None:
            raise UHIRParseError("bind µhIR must declare schedule kind=...")
        if not design.resources:
            raise UHIRParseError("bind µhIR must declare resources")
        if design.controllers:
            raise UHIRParseError("bind µhIR must not declare controllers")
    elif design.stage == "fsm":
        if design.schedule is None:
            raise UHIRParseError("fsm µhIR must declare schedule kind=...")
        if not design.resources:
            raise UHIRParseError("fsm µhIR must declare resources")
        if not design.controllers:
            raise UHIRParseError("fsm µhIR must declare at least one controller")
    elif design.controllers:
        raise UHIRParseError(f"{design.stage} µhIR must not declare controllers")

    controller_names = {controller.name for controller in design.controllers}
    if len(controller_names) != len(design.controllers):
        raise UHIRParseError("controller identifiers must be unique within one µhIR design")
    for controller in design.controllers:
        _validate_controller(controller, design.stage)

    region_ids = {region.id for region in design.regions}
    if len(region_ids) != len(design.regions):
        raise UHIRParseError("region identifiers must be unique within one µhIR design")

    node_ids: set[str] = set()
    resource_ids = {resource.id for resource in design.resources}
    resource_kinds = {resource.id: resource.kind for resource in design.resources}
    known_endpoints = set(region_ids)
    exg_vertex_partitions: dict[str, tuple[str, str]] = {}
    exg_functional_units: set[str] = set()
    exg_operations: set[str] = set()
    used_canonical_operations: set[str] = set()

    for region in design.regions:
        region_stage = _region_stage(design.stage, region)
        if region.parent is not None and region.parent not in region_ids:
            raise UHIRParseError(f"region '{region.id}' references unknown parent '{region.parent}'")
        if region_stage == "exg" and region.parent is not None:
            raise UHIRParseError(f"exg µhIR region '{region.id}' must not declare a parent")
        for region_ref in region.region_refs:
            if region_ref.target not in region_ids:
                raise UHIRParseError(f"region '{region.id}' references unknown region '{region_ref.target}'")
        if region_stage == "exg" and region.region_refs:
            raise UHIRParseError(f"exg µhIR region '{region.id}' must not reference child regions")
        for node in region.nodes:
            if node.id in node_ids:
                raise UHIRParseError(f"node identifier '{node.id}' is duplicated")
            node_ids.add(node.id)
            known_endpoints.add(node.id)
            _validate_node_for_stage(node, region_stage)
            if region_stage == "exg":
                partition = node.attributes["partition"]
                resolved_name = _exg_vertex_name(node)
                exg_vertex_partitions[node.id] = (resolved_name, partition)
                if partition == "fu":
                    if resolved_name in exg_operations:
                        raise UHIRParseError(
                            f"executability graph is not bipartite: shared FU/op vertex '{resolved_name}'"
                        )
                    exg_functional_units.add(resolved_name)
                else:
                    if resolved_name in exg_functional_units:
                        raise UHIRParseError(
                            f"executability graph is not bipartite: shared FU/op vertex '{resolved_name}'"
                        )
                    exg_operations.add(resolved_name)
            elif design.stage == "alloc":
                operation = _canonical_operation_name(node.opcode)
                if operation is not None:
                    used_canonical_operations.add(operation)
            if design.stage in {"bind", "fsm"}:
                bind_target = node.attributes.get("bind")
                class_name = node.attributes.get("class")
                if class_name not in {"CTRL", "ADAPT"}:
                    if not isinstance(bind_target, str) or bind_target not in resource_ids:
                        raise UHIRParseError(f"{design.stage}-stage node '{node.id}' must reference a declared resource")
                elif bind_target is not None and (not isinstance(bind_target, str) or bind_target not in resource_ids):
                    raise UHIRParseError(f"{design.stage}-stage node '{node.id}' must reference a declared resource")
        if region_stage in {"seq", "alloc", "exg"}:
            if region.steps is not None or region.latency is not None or region.initiation_interval is not None:
                raise UHIRParseError(f"{region_stage} µhIR region '{region.id}' must not contain schedule summaries")
            if region.value_bindings or region.muxes:
                raise UHIRParseError(f"{region_stage} µhIR region '{region.id}' must not contain bind-only statements")
        if region_stage == "sched":
            if region.steps is not None:
                start, end = region.steps
                if not isinstance(start, (int, TimingExpr)) or not isinstance(end, (int, TimingExpr)):
                    raise UHIRParseError(f"sched µhIR region '{region.id}' has invalid steps summary")
                if isinstance(start, int) and isinstance(end, int) and start > end:
                    raise UHIRParseError(f"sched µhIR region '{region.id}' has steps with start>end")
            if region.latency is not None and not isinstance(region.latency, (int, TimingExpr)):
                raise UHIRParseError(f"sched µhIR region '{region.id}' has invalid latency summary")
            if region.initiation_interval is not None and not isinstance(region.initiation_interval, (int, TimingExpr)):
                raise UHIRParseError(f"sched µhIR region '{region.id}' has invalid ii summary")
        if region_stage == "bind":
            if region.steps is not None:
                start, end = region.steps
                if not isinstance(start, (int, TimingExpr)) or not isinstance(end, (int, TimingExpr)):
                    raise UHIRParseError(f"bind µhIR region '{region.id}' has invalid steps summary")
                if isinstance(start, int) and isinstance(end, int) and start > end:
                    raise UHIRParseError(f"bind µhIR region '{region.id}' has steps with start>end")
            if region.latency is not None and not isinstance(region.latency, (int, TimingExpr)):
                raise UHIRParseError(f"bind µhIR region '{region.id}' has invalid latency summary")
            if region.initiation_interval is not None and not isinstance(region.initiation_interval, (int, TimingExpr)):
                raise UHIRParseError(f"bind µhIR region '{region.id}' has invalid ii summary")
        if region_stage == "fsm":
            if region.steps is not None:
                start, end = region.steps
                if not isinstance(start, int) or not isinstance(end, int):
                    raise UHIRParseError(f"fsm µhIR region '{region.id}' must use integer steps summary")
                if start > end:
                    raise UHIRParseError(f"fsm µhIR region '{region.id}' has steps with start>end")
            if region.latency is not None and not isinstance(region.latency, int):
                raise UHIRParseError(f"fsm µhIR region '{region.id}' must use integer latency")
            if region.initiation_interval is not None and not isinstance(region.initiation_interval, int):
                raise UHIRParseError(f"fsm µhIR region '{region.id}' must use integer ii")
        if region_stage == "exg" and region.mappings:
            raise UHIRParseError(f"exg µhIR region '{region.id}' must not contain source mappings")
        if design.stage == "sched" and (region.value_bindings or region.muxes):
            raise UHIRParseError(f"sched µhIR region '{region.id}' must not contain bind-only statements")
        local_value_ids = {mapping.source_id for mapping in region.mappings}
        local_value_ids.update(node.id for node in region.nodes)
        for value_binding in region.value_bindings:
            if value_binding.register not in resource_ids:
                raise UHIRParseError(
                    f"value binding '{value_binding.producer} -> {value_binding.register}' references unknown register"
                )
            if resource_kinds[value_binding.register] != "reg":
                raise UHIRParseError(
                    f"value binding '{value_binding.producer} -> {value_binding.register}' must target a reg resource"
                )
            if value_binding.producer not in local_value_ids:
                raise UHIRParseError(
                    f"value binding '{value_binding.producer} -> {value_binding.register}' must reference one mapped value id or local node id"
                )
        for mux in region.muxes:
            if mux.output not in resource_ids:
                raise UHIRParseError(f"mux '{mux.id}' references unknown output '{mux.output}'")
            if resource_kinds[mux.output] != "reg":
                raise UHIRParseError(f"mux '{mux.id}' must target a reg resource output")

    exg_connected_operations: set[str] = set()
    for region in design.regions:
        region_stage = _region_stage(design.stage, region)
        for edge in region.edges:
            if edge.source not in known_endpoints:
                raise UHIRParseError(f"edge in region '{region.id}' references unknown source '{edge.source}'")
            if edge.target not in known_endpoints:
                raise UHIRParseError(f"edge in region '{region.id}' references unknown target '{edge.target}'")
            if region_stage == "exg":
                if edge.kind != "exg":
                    raise UHIRParseError(f"exg µhIR edge '{edge.source} -- {edge.target}' must use kind 'exg'")
                if edge.directed:
                    raise UHIRParseError(f"exg µhIR edge '{edge.source} -- {edge.target}' must be undirected")
                if edge.source not in node_ids or edge.target not in node_ids:
                    raise UHIRParseError(f"exg µhIR edge '{edge.source} -- {edge.target}' must connect graph nodes")
                ii = edge.attributes.get("ii")
                delay = edge.attributes.get("d")
                if not isinstance(ii, int) or not isinstance(delay, int):
                    raise UHIRParseError(f"exg µhIR edge '{edge.source} -- {edge.target}' must define integer ii/d weights")
                if ii > delay:
                    raise UHIRParseError(f"exg µhIR edge '{edge.source} -- {edge.target}' violates ii<=d: ii={ii}, d={delay}")
                source_name, source_partition = exg_vertex_partitions[edge.source]
                target_name, target_partition = exg_vertex_partitions[edge.target]
                if source_partition == target_partition:
                    raise UHIRParseError(
                        f"executability graph is not bipartite: edge '{edge.source} -- {edge.target}' must connect one FU and one op"
                    )
                if source_partition == "op":
                    exg_connected_operations.add(source_name)
                else:
                    exg_connected_operations.add(target_name)
        for mapping in region.mappings:
            if mapping.node_id not in node_ids:
                raise UHIRParseError(f"map in region '{region.id}' references unknown node '{mapping.node_id}'")

    disconnected_canonical_ops = sorted(
        operation for operation in exg_operations if operation in _UIR_LANGUAGE_OPCODES and operation not in exg_connected_operations
    )
    if disconnected_canonical_ops:
        names = ", ".join(disconnected_canonical_ops)
        raise UHIRParseError(f"executability graph leaves canonical µIR operations disconnected: {names}")
    if design.stage == "alloc" and exg_operations:
        missing_embedded_ops = sorted(operation for operation in used_canonical_operations if operation not in exg_connected_operations)
        if missing_embedded_ops:
            names = ", ".join(missing_embedded_ops)
            raise UHIRParseError(f"embedded executability graph does not cover canonical µIR operations used in alloc µhIR: {names}")


def _region_stage(design_stage: str, region: UHIRRegion) -> str:
    if design_stage == "alloc" and region.kind == "executability":
        return "exg"
    return design_stage


def _validate_node_for_stage(node: UHIRNode, stage: str) -> None:
    if stage == "exg":
        partition = node.attributes.get("partition")
        if partition not in {"fu", "op"}:
            raise UHIRParseError(f"exg-stage node '{node.id}' must declare partition=fu|op")
        if node.opcode != partition:
            raise UHIRParseError(f"exg-stage node '{node.id}' must use opcode '{partition}'")
        if node.operands:
            raise UHIRParseError(f"exg-stage node '{node.id}' must not declare operands")
        if node.result_type is not None:
            raise UHIRParseError(f"exg-stage node '{node.id}' must not declare a result type")
        vertex_name = node.attributes.get("name")
        if vertex_name is not None and (not isinstance(vertex_name, str) or not vertex_name):
            raise UHIRParseError(f"exg-stage node '{node.id}' has invalid name=... attribute")
        resolved_name = node.id if vertex_name is None else vertex_name
        if partition == "fu" and resolved_name != resolved_name.upper():
            raise UHIRParseError(f"exg-stage FU node '{node.id}' must use a CAPITALIZED vertex name")
        if partition == "op" and resolved_name != resolved_name.lower():
            raise UHIRParseError(f"exg-stage op node '{node.id}' must use a lower-case vertex name")
        forbidden = [name for name in ("class", "ii", "delay", "start", "end", "bind") if name in node.attributes]
        if forbidden:
            raise UHIRParseError(f"exg-stage node '{node.id}' contains forbidden attributes: {', '.join(forbidden)}")
        return

    pred = node.attributes.get("pred")
    if pred is not None and (not isinstance(pred, str) or not pred):
        raise UHIRParseError(f"node '{node.id}' has invalid pred=... attribute")

    incoming = node.attributes.get("incoming")
    if incoming is not None:
        if node.opcode != "phi":
            raise UHIRParseError(f"node '{node.id}' may only use incoming=[...] when opcode is 'phi'")
        if not isinstance(incoming, tuple) or len(incoming) != len(node.operands):
            raise UHIRParseError(f"phi node '{node.id}' must use incoming=[...] with one label per operand")
        if any(not isinstance(label, str) or not label for label in incoming):
            raise UHIRParseError(f"phi node '{node.id}' has invalid incoming=[...] labels")

    if node.opcode == "sel" and len(node.operands) != 3:
        raise UHIRParseError(f"sel node '{node.id}' must declare exactly three operands: cond, true_value, false_value")

    for name in ("header_label", "true_label", "false_label", "true_input_label", "false_input_label"):
        value = node.attributes.get(name)
        if value is None:
            continue
        if node.opcode != "branch":
            raise UHIRParseError(f"node '{node.id}' may only use {name}=... when opcode is 'branch'")
        if not isinstance(value, str) or not value:
            raise UHIRParseError(f"branch node '{node.id}' has invalid {name}=... attribute")

    class_name = node.attributes.get("class")
    initiation_interval = node.attributes.get("ii")
    delay = node.attributes.get("delay")
    if stage in {"sched", "bind"}:
        for name in ("ii", "delay", "start", "end", "iter_latency", "iter_initiation_interval", "iter_ramp_down"):
            value = node.attributes.get(name)
            if isinstance(value, str):
                try:
                    node.attributes[name] = parse_timing_expr(value)
                except ValueError as exc:
                    raise UHIRParseError(f"{stage}-stage node '{node.id}' has invalid {name}=... timing expression") from exc
        initiation_interval = node.attributes.get("ii")
        delay = node.attributes.get("delay")
    if stage in {"alloc", "sched", "bind", "fsm"}:
        if not isinstance(class_name, str) or not class_name:
            raise UHIRParseError(f"node '{node.id}' is missing class=...")
    if stage == "alloc":
        if not isinstance(initiation_interval, int):
            raise UHIRParseError(f"node '{node.id}' is missing ii=...")
    if stage in {"alloc", "fsm"}:
        if not isinstance(delay, int):
            raise UHIRParseError(f"node '{node.id}' is missing delay=...")
    if stage == "bind":
        if not isinstance(delay, (int, TimingExpr)):
            raise UHIRParseError(f"bind-stage node '{node.id}' is missing delay=...")
    if stage == "sched":
        if not isinstance(delay, (int, TimingExpr)):
            raise UHIRParseError(f"sched-stage node '{node.id}' is missing delay=...")
    if stage == "bind":
        start = node.attributes.get("start")
        end = node.attributes.get("end")
        if not isinstance(start, (int, TimingExpr)) or not isinstance(end, (int, TimingExpr)):
            raise UHIRParseError(f"bind-stage node '{node.id}' is missing start/end attributes")
        if isinstance(start, int) and isinstance(end, int) and end < start:
            raise UHIRParseError(f"bind-stage node '{node.id}' has end < start")
    if stage == "fsm":
        start = node.attributes.get("start")
        end = node.attributes.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise UHIRParseError(f"{stage}-stage node '{node.id}' is missing start/end attributes")
        if end < start:
            raise UHIRParseError(f"{stage}-stage node '{node.id}' has end < start")
    if stage == "sched":
        start = node.attributes.get("start")
        end = node.attributes.get("end")
        if not isinstance(start, (int, TimingExpr)) or not isinstance(end, (int, TimingExpr)):
            raise UHIRParseError(f"{stage}-stage node '{node.id}' is missing start/end attributes")
        if isinstance(start, int) and isinstance(end, int) and end < start:
            raise UHIRParseError(f"{stage}-stage node '{node.id}' has end < start")
    if stage in {"seq", "alloc"}:
        forbidden = [name for name in ("start", "end", "bind") if name in node.attributes]
        if forbidden:
            raise UHIRParseError(f"{stage}-stage node '{node.id}' contains forbidden attributes: {', '.join(forbidden)}")


def _validate_controller(controller: UHIRController, stage: str) -> None:
    if stage != "fsm":
        raise UHIRParseError(f"{stage} µhIR must not declare controllers")
    encoding = controller.attributes.get("encoding")
    if encoding not in {"binary", "one_hot"}:
        raise UHIRParseError(f"controller '{controller.name}' must declare encoding=binary|one_hot")
    protocol = controller.attributes.get("protocol")
    if protocol not in {"req_resp", "act_done"}:
        raise UHIRParseError(f"controller '{controller.name}' must declare protocol=req_resp|act_done")
    completion_order = controller.attributes.get("completion_order")
    if completion_order != "in_order":
        raise UHIRParseError(f"controller '{controller.name}' must declare completion_order=in_order")
    overlap = controller.attributes.get("overlap")
    if overlap is not True:
        raise UHIRParseError(f"controller '{controller.name}' must declare overlap=true")
    if protocol == "req_resp":
        expected_inputs = [("req_valid", "i1"), ("resp_ready", "i1")]
        expected_outputs = [("req_ready", "i1"), ("resp_valid", "i1")]
        input_error = f"controller '{controller.name}' must declare inputs: req_valid:i1, resp_ready:i1"
        output_error = f"controller '{controller.name}' must declare outputs: req_ready:i1, resp_valid:i1"
    else:
        expected_inputs = [("act_valid", "i1"), ("done_ready", "i1")]
        expected_outputs = [("act_ready", "i1"), ("done_valid", "i1")]
        input_error = f"controller '{controller.name}' must declare inputs: act_valid:i1, done_ready:i1"
        output_error = f"controller '{controller.name}' must declare outputs: act_ready:i1, done_valid:i1"
    if [(port.name, port.type) for port in controller.inputs] != expected_inputs:
        raise UHIRParseError(input_error)
    if [(port.name, port.type) for port in controller.outputs] != expected_outputs:
        raise UHIRParseError(output_error)
    state_names = {state.name for state in controller.states}
    if len(state_names) != len(controller.states):
        raise UHIRParseError(f"controller '{controller.name}' must use unique state names")
    for transition in controller.transitions:
        if transition.source not in state_names or transition.target not in state_names:
            raise UHIRParseError(
                f"controller '{controller.name}' transition '{transition.source} -> {transition.target}' references an unknown state"
            )
    for emit in controller.emits:
        if emit.state not in state_names:
            raise UHIRParseError(f"controller '{controller.name}' emit for '{emit.state}' references an unknown state")
    link_children = {link.child for link in controller.links}
    if len(link_children) != len(controller.links):
        raise UHIRParseError(f"controller '{controller.name}' must use unique child-controller link targets")


def _exg_vertex_name(node: UHIRNode) -> str:
    vertex_name = node.attributes.get("name")
    return node.id if vertex_name is None else vertex_name


def _canonical_operation_name(opcode: str) -> str | None:
    normalized = opcode.lower()
    if normalized in _UIR_LANGUAGE_OPCODES:
        return normalized
    return None
