"""Parser helpers for textual µglIR artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from .model import (
    UGLIRAddressMap,
    UGLIRAddressMapEntry,
    UGLIRAssign,
    UGLIRAttach,
    UGLIRConstant,
    UGLIRDesign,
    UGLIRMux,
    UGLIRMuxCase,
    UGLIRPort,
    UGLIRResource,
    UGLIRSeqBlock,
    UGLIRSeqUpdate,
)

_IDENT_RE = r"[A-Za-z_][\w$]*"
_RESOURCE_VALUE_RE = r"[A-Za-z_][\w$]*(?:<[^>\n]+>)?"
_DESIGN_RE = re.compile(rf"^design\s+({_IDENT_RE})$")
_STAGE_RE = re.compile(r"^stage\s+(uglir)$")
_PORT_RE = re.compile(rf"^(input|output)\s+({_IDENT_RE})\s*:\s*(.+)$")
_CONST_RE = re.compile(rf"^const\s+({_IDENT_RE})\s*=\s*(.+?)\s*:\s*(.+)$")
_ADDRESS_MAP_START_RE = re.compile(rf"^address_map\s+({_IDENT_RE})\s*\{{$")
_ADDRESS_MAP_ENTRY_RE = re.compile(rf"^(register|memory)\s+({_IDENT_RE})(?:\s+(.*))?$")
_REG_RE = re.compile(rf"^reg\s+({_IDENT_RE})\s*:\s*([A-Za-z0-9_<>\[\]]+)$")
_NET_RE = re.compile(rf"^net\s+({_IDENT_RE})\s*:\s*([A-Za-z0-9_<>\[\]]+)$")
_MEM_RE = re.compile(rf"^mem\s+({_IDENT_RE})\s*:\s*([A-Za-z0-9_<>\[\]]+)$")
_INST_RE = re.compile(rf"^inst\s+({_IDENT_RE})\s*:\s*({_RESOURCE_VALUE_RE})$")
_MUX_RESOURCE_RE = re.compile(rf"^mux\s+({_IDENT_RE})\s*:\s*([A-Za-z0-9_<>\[\]]+)$")
_PORT_RESOURCE_RE = re.compile(rf"^port\s+({_IDENT_RE})\s*:\s*({_RESOURCE_VALUE_RE})(?:\s+({_IDENT_RE}))?$")
_ASSIGN_RE = re.compile(rf"^assign\s+({_IDENT_RE})\s*=\s*(.+)$")
_ATTACH_RE = re.compile(rf"^({_IDENT_RE})\.({_IDENT_RE})\(([_A-Za-z][\w$]*)\)$")
_UGLIR_MUX_START_RE = re.compile(rf"^mux\s+({_IDENT_RE})\s*:\s*(.+?)\s+sel=(\S+)\s*\{{$")
_UGLIR_MUX_CASE_RE = re.compile(r"^(\S+)\s*->\s*(\S+)$")
_SEQ_START_RE = re.compile(rf"^seq\s+({_IDENT_RE})\s*\{{$")
_IF_RE = re.compile(r"^if\s+(.+)\s*\{$")
_UPDATE_TARGET_RE = rf"{_IDENT_RE}(?:\[(.+)\])?"
_UPDATE_RE = re.compile(rf"^({_UPDATE_TARGET_RE})\s*<=\s*(.+)$")
_UGLIR_EXPR_IDENT_RE = re.compile(rf"\b({_IDENT_RE})\b")


class UGLIRParseError(ValueError):
    """Raised when textual µglIR cannot be parsed."""


def parse_uglir(text: str) -> UGLIRDesign:
    """Parse one textual µglIR design."""
    lines = _normalize_lines(text)
    if not lines:
        raise UGLIRParseError("empty µglIR input")

    index = 0
    name = _parse_design(lines[index], 1)
    index += 1
    if index >= len(lines):
        raise UGLIRParseError("missing stage declaration")
    stage = _parse_stage(lines[index], index + 1)
    index += 1

    design = UGLIRDesign(name=name, stage=stage)
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
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
        if line.startswith("address_map "):
            address_map, index = _parse_address_map(lines, index)
            design.address_maps.append(address_map)
            continue
        if line == "resources {":
            if design.resources:
                raise UGLIRParseError(f"duplicate resources block at line {line_number}")
            resources, index = _parse_resources_block(lines, index)
            design.resources.extend(resources)
            continue
        if match := _ASSIGN_RE.fullmatch(line):
            design.assigns.append(UGLIRAssign(match.group(1), _unwrap_parenthesized_expr(match.group(2).strip())))
            index += 1
            continue
        if match := _ATTACH_RE.fullmatch(line):
            design.attachments.append(UGLIRAttach(match.group(1), match.group(2), match.group(3)))
            index += 1
            continue
        if line.startswith("mux "):
            mux, index = _parse_mux(lines, index)
            design.muxes.append(mux)
            continue
        if line.startswith("seq "):
            seq_block, index = _parse_seq_block(lines, index)
            design.seq_blocks.append(seq_block)
            continue
        raise UGLIRParseError(f"unsupported top-level µglIR syntax at line {line_number}: {line!r}")

    _validate_uglir_design(design)
    return design


def parse_uglir_file(path: str | Path) -> UGLIRDesign:
    """Parse one textual µglIR design from disk."""
    return parse_uglir(Path(path).read_text(encoding="utf-8"))


def _normalize_lines(text: str) -> list[str]:
    lines: list[str] = []
    buffer: list[str] = []
    paren_depth = 0
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
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
        raise UGLIRParseError(f"expected design declaration at line {line_number}: {line!r}")
    return match.group(1)


def _parse_stage(line: str, line_number: int) -> str:
    match = _STAGE_RE.fullmatch(line)
    if match is None:
        raise UGLIRParseError(f"expected µglIR stage declaration at line {line_number}: {line!r}")
    return match.group(1)


def _try_parse_port(line: str) -> UGLIRPort | None:
    match = _PORT_RE.fullmatch(line)
    if match is None:
        return None
    direction, name, type_hint = match.groups()
    return UGLIRPort(direction, name, type_hint.strip())


def _try_parse_const(line: str) -> UGLIRConstant | None:
    match = _CONST_RE.fullmatch(line)
    if match is None:
        return None
    name, value_text, type_hint = match.groups()
    return UGLIRConstant(name, _parse_scalar_text(_unwrap_parenthesized_expr(value_text.strip())), type_hint.strip())


def _parse_resources_block(lines: list[str], index: int) -> tuple[list[UGLIRResource], int]:
    resources: list[UGLIRResource] = []
    index += 1
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        if line == "}":
            return resources, index + 1
        if match := _REG_RE.fullmatch(line):
            resources.append(UGLIRResource("reg", match.group(1), match.group(2)))
        elif match := _NET_RE.fullmatch(line):
            resources.append(UGLIRResource("net", match.group(1), match.group(2)))
        elif match := _MEM_RE.fullmatch(line):
            resources.append(UGLIRResource("mem", match.group(1), match.group(2)))
        elif match := _INST_RE.fullmatch(line):
            resources.append(UGLIRResource("inst", match.group(1), match.group(2)))
        elif match := _MUX_RESOURCE_RE.fullmatch(line):
            resources.append(UGLIRResource("mux", match.group(1), match.group(2)))
        elif match := _PORT_RESOURCE_RE.fullmatch(line):
            resources.append(UGLIRResource("port", match.group(1), match.group(2), match.group(3)))
        else:
            raise UGLIRParseError(f"invalid µglIR resources declaration at line {line_number}: {line!r}")
        index += 1
    raise UGLIRParseError("unterminated µglIR resources block")


def _parse_address_map(lines: list[str], index: int) -> tuple[UGLIRAddressMap, int]:
    line = lines[index]
    line_number = index + 1
    match = _ADDRESS_MAP_START_RE.fullmatch(line)
    if match is None:
        raise UGLIRParseError(f"invalid address-map declaration at line {line_number}: {line!r}")
    address_map = UGLIRAddressMap(match.group(1))
    index += 1
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        if line == "}":
            return address_map, index + 1
        match = _ADDRESS_MAP_ENTRY_RE.fullmatch(line)
        if match is None:
            raise UGLIRParseError(f"invalid address-map entry at line {line_number}: {line!r}")
        attributes = {} if match.group(3) is None else _parse_attrs(match.group(3))
        address_map.entries.append(UGLIRAddressMapEntry(match.group(1), match.group(2), attributes))
        index += 1
    raise UGLIRParseError(f"unterminated address_map '{address_map.name}'")


def _parse_mux(lines: list[str], index: int) -> tuple[UGLIRMux, int]:
    line = lines[index]
    line_number = index + 1
    match = _UGLIR_MUX_START_RE.fullmatch(line)
    if match is None:
        raise UGLIRParseError(f"invalid µglIR mux declaration at line {line_number}: {line!r}")
    mux = UGLIRMux(match.group(1), match.group(2).strip(), match.group(3).strip())
    index += 1
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        if line == "}":
            return mux, index + 1
        match = _UGLIR_MUX_CASE_RE.fullmatch(line)
        if match is None:
            raise UGLIRParseError(f"invalid µglIR mux case at line {line_number}: {line!r}")
        mux.cases.append(UGLIRMuxCase(match.group(1), match.group(2)))
        index += 1
    raise UGLIRParseError(f"unterminated µglIR mux '{mux.name}'")


def _parse_seq_block(lines: list[str], index: int) -> tuple[UGLIRSeqBlock, int]:
    line = lines[index]
    line_number = index + 1
    match = _SEQ_START_RE.fullmatch(line)
    if match is None:
        raise UGLIRParseError(f"invalid µglIR seq declaration at line {line_number}: {line!r}")
    seq_block = UGLIRSeqBlock(clock=match.group(1))
    index += 1
    if index >= len(lines):
        raise UGLIRParseError(f"unterminated seq block '{seq_block.clock}'")
    line = lines[index]
    if_match = _IF_RE.fullmatch(line)
    if if_match is not None:
        seq_block.reset = _unwrap_parenthesized_expr(if_match.group(1).strip())
        index += 1
        seq_block.reset_updates, index = _parse_seq_updates_until(lines, index, "}")
        if index >= len(lines):
            raise UGLIRParseError(f"seq block '{seq_block.clock}' is missing else {{")
        if lines[index] == "} else {":
            index += 1
        elif lines[index] == "}":
            index += 1
            if index >= len(lines) or lines[index] != "else {":
                raise UGLIRParseError(f"seq block '{seq_block.clock}' is missing else {{")
            index += 1
        else:
            raise UGLIRParseError(f"seq block '{seq_block.clock}' is missing else {{")
        seq_block.updates, index = _parse_seq_updates_with_guards_until(lines, index, "}")
        if index >= len(lines) or lines[index] != "}":
            raise UGLIRParseError(f"unterminated seq block '{seq_block.clock}'")
        index += 1
        if index >= len(lines) or lines[index] != "}":
            raise UGLIRParseError(f"unterminated seq block '{seq_block.clock}'")
        return seq_block, index + 1
    seq_block.updates, index = _parse_seq_updates_with_guards_until(lines, index, "}")
    return seq_block, index + 1


def _parse_seq_updates_until(lines: list[str], index: int, terminator: str) -> tuple[list[UGLIRSeqUpdate], int]:
    updates: list[UGLIRSeqUpdate] = []
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        if line == terminator or line == f"{terminator} else {{":
            return updates, index
        match = _UPDATE_RE.fullmatch(line)
        if match is None:
            raise UGLIRParseError(f"invalid sequential update at line {line_number}: {line!r}")
        updates.append(UGLIRSeqUpdate(match.group(1), _unwrap_parenthesized_expr(match.group(3).strip())))
        index += 1
    raise UGLIRParseError("unterminated sequential update block")


def _parse_seq_updates_with_guards_until(lines: list[str], index: int, terminator: str) -> tuple[list[UGLIRSeqUpdate], int]:
    updates: list[UGLIRSeqUpdate] = []
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        if line == terminator:
            return updates, index
        guard_match = _IF_RE.fullmatch(line)
        if guard_match is not None:
            guard = _unwrap_parenthesized_expr(guard_match.group(1).strip())
            index += 1
            guarded_updates, index = _parse_seq_updates_until(lines, index, "}")
            updates.extend(UGLIRSeqUpdate(update.target, update.value, guard) for update in guarded_updates)
            index += 1
            continue
        match = _UPDATE_RE.fullmatch(line)
        if match is None:
            raise UGLIRParseError(f"invalid sequential update at line {line_number}: {line!r}")
        updates.append(UGLIRSeqUpdate(match.group(1), _unwrap_parenthesized_expr(match.group(3).strip())))
        index += 1
    raise UGLIRParseError("unterminated sequential update block")


def _parse_attrs(text: str) -> dict[str, str | int | bool | tuple[str, ...]]:
    attrs: dict[str, str | int | bool | tuple[str, ...]] = {}
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
            raise UGLIRParseError(f"invalid attribute spelling {text!r}")
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


def _parse_attr_value(text: str) -> str | int | bool | tuple[str, ...]:
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


def _validate_uglir_design(design: UGLIRDesign) -> None:
    if design.stage != "uglir":
        raise UGLIRParseError(f"µglIR parser expects stage 'uglir', got stage '{design.stage}'")
    if not design.resources:
        raise UGLIRParseError("µglIR design must declare resources")

    resource_ids = {resource.id for resource in design.resources}
    resource_kinds = {resource.id: resource.kind for resource in design.resources}
    signal_names = {port.name for port in design.inputs + design.outputs}
    signal_names.update(resource.id for resource in design.resources if resource.kind in {"reg", "net", "mux", "mem"})
    instance_ids = {resource.id for resource in design.resources if resource.kind == "inst"}
    mux_ids = {resource.id for resource in design.resources if resource.kind == "mux"}

    for attachment in design.attachments:
        if attachment.instance not in instance_ids:
            raise UGLIRParseError(f"µglIR attachment '{attachment.instance}.{attachment.port}(...)' references unknown instance")
        if attachment.signal not in signal_names:
            raise UGLIRParseError(f"µglIR attachment '{attachment.instance}.{attachment.port}({attachment.signal})' references unknown signal")
    for assign in design.assigns:
        if assign.target not in signal_names:
            raise UGLIRParseError(f"µglIR assign '{assign.target} = ...' references unknown target signal")
    mux_names = {mux.name for mux in design.muxes}
    if len(mux_names) != len(design.muxes):
        raise UGLIRParseError("µglIR mux names must be unique")
    for mux in design.muxes:
        if mux.name not in mux_ids:
            raise UGLIRParseError(f"µglIR mux '{mux.name}' must have a declared mux resource")
        if mux.select not in signal_names:
            raise UGLIRParseError(f"µglIR mux '{mux.name}' references unknown select signal '{mux.select}'")
        if not mux.cases:
            raise UGLIRParseError(f"µglIR mux '{mux.name}' must declare at least one case")
        for case in mux.cases:
            if case.source not in signal_names:
                raise UGLIRParseError(f"µglIR mux '{mux.name}' case '{case.key}' references unknown source '{case.source}'")
    input_names = {port.name for port in design.inputs}
    for seq_block in design.seq_blocks:
        if seq_block.clock not in input_names:
            raise UGLIRParseError(f"µglIR seq block clock '{seq_block.clock}' must be an input port")
        if seq_block.reset is not None:
            _validate_uglir_expr_identifiers(seq_block.reset, signal_names, f"µglIR seq block reset '{seq_block.reset}'")
        for update in [*seq_block.reset_updates, *seq_block.updates]:
            target_base, _ = _split_seq_target(update.target)
            if target_base not in resource_ids or resource_kinds[target_base] not in {"reg", "mem"}:
                raise UGLIRParseError(f"µglIR sequential update target '{update.target}' must be a reg or mem resource")
            if update.enable is not None:
                _validate_uglir_expr_identifiers(update.enable, signal_names, f"µglIR sequential enable '{update.enable}'")


def _split_seq_target(target: str) -> tuple[str, str | None]:
    match = re.fullmatch(rf"({_IDENT_RE})(?:\[(.+)\])?", target)
    if match is None:
        raise UGLIRParseError(f"µglIR sequential update target '{target}' must be an identifier or indexed memory element")
    return match.group(1), match.group(2)


def _validate_uglir_expr_identifiers(expr: str, signal_names: set[str], prefix: str) -> None:
    allowed = signal_names | {"true", "false"}
    for ident in _UGLIR_EXPR_IDENT_RE.findall(expr):
        if ident not in allowed:
            raise UGLIRParseError(f"{prefix} references unknown signal")


UGLIRParseError = UGLIRParseError

__all__ = ["UGLIRParseError", "parse_uglir", "parse_uglir_file"]
