"""Foreign HDL import helpers for shared component libraries."""

from __future__ import annotations

import re
from pathlib import Path


def import_verilog_component_stub(
    *,
    verilog_text: str,
    module_name: str,
    source_path: Path | None = None,
    ops: tuple[str, ...] = (),
    kind: str = "combinational",
) -> dict[str, object]:
    """Build one component-library stub from one Verilog module interface."""
    parameter_block, port_block, module_body = _extract_verilog_module_header(verilog_text, module_name)
    parameters = _parse_verilog_parameters(parameter_block)
    ports = _parse_verilog_ports(port_block, module_body=module_body)
    return {
        "kind": kind,
        "parameters": parameters,
        "hdl": _hdl_linkage_payload(module_name, () if source_path is None else (source_path,)),
        "ports": ports,
        "supports": {op: _support_stub() for op in ops},
    }


def import_verilog_component_stub_from_files(
    *,
    source_files: tuple[tuple[Path, str], ...],
    module_name: str,
    ops: tuple[str, ...] = (),
    kind: str = "combinational",
) -> dict[str, object]:
    """Build one component-library stub from one target module plus related source files."""
    if not source_files:
        raise ValueError("verilog import requires at least one source file")
    related_paths, target_text = _resolve_related_verilog_sources(source_files, module_name)
    parameter_block, port_block, module_body = _extract_verilog_module_header(target_text, module_name)
    parameters = _parse_verilog_parameters(parameter_block)
    ports = _parse_verilog_ports(port_block, module_body=module_body)
    return {
        "kind": kind,
        "parameters": parameters,
        "hdl": _hdl_linkage_payload(module_name, related_paths),
        "ports": ports,
        "supports": {op: _support_stub() for op in ops},
    }


def _support_stub() -> dict[str, object]:
    return {
        "ii": "TODO",
        "d": "TODO",
        "bind": "TODO",
    }


def _hdl_linkage_payload(module_name: str, source_paths: tuple[Path, ...]) -> dict[str, object]:
    hdl: dict[str, object] = {
        "language": "verilog",
        "module": module_name,
    }
    if len(source_paths) == 1:
        hdl["source"] = str(source_paths[0])
    elif len(source_paths) > 1:
        hdl["sources"] = [str(path) for path in source_paths]
    return hdl


def _resolve_related_verilog_sources(
    source_files: tuple[tuple[Path, str], ...],
    module_name: str,
) -> tuple[tuple[Path, ...], str]:
    module_to_file: dict[str, Path] = {}
    module_bodies: dict[str, str] = {}
    module_texts: dict[str, str] = {}
    for source_path, verilog_text in source_files:
        for discovered_name, module_body in _iter_verilog_modules(verilog_text):
            if discovered_name in module_to_file:
                raise ValueError(
                    f"verilog import found duplicate definition for module '{discovered_name}' in "
                    f"'{module_to_file[discovered_name]}' and '{source_path}'"
                )
            module_to_file[discovered_name] = source_path
            module_bodies[discovered_name] = module_body
            module_texts[discovered_name] = verilog_text
    target_text = module_texts.get(module_name)
    if target_text is None:
        raise ValueError(f"verilog source set does not define module '{module_name}'")
    known_modules = frozenset(module_to_file)
    related_modules: set[str] = set()
    worklist = [module_name]
    while worklist:
        current = worklist.pop()
        if current in related_modules:
            continue
        related_modules.add(current)
        current_body = module_bodies.get(current, "")
        for dependency in _collect_instantiated_modules(current_body, known_modules):
            if dependency not in related_modules:
                worklist.append(dependency)
    related_paths = tuple(
        source_path
        for source_path, _text in source_files
        if any(module_to_file[name] == source_path for name in related_modules)
    )
    return related_paths, target_text


def _iter_verilog_modules(verilog_text: str) -> list[tuple[str, str]]:
    modules: list[tuple[str, str]] = []
    pattern = re.compile(r"\bmodule\s+([A-Za-z_][\w$]*)\b", re.MULTILINE)
    cursor = 0
    while True:
        match = pattern.search(verilog_text, cursor)
        if match is None:
            return modules
        module_name = match.group(1)
        end_match = re.search(r"\bendmodule\b", verilog_text[match.end() :], re.MULTILINE)
        if end_match is None:
            raise ValueError(f"verilog source module '{module_name}' is missing endmodule")
        block_end = match.end() + end_match.end()
        module_text = verilog_text[match.start() : block_end]
        _parameter_block, _port_block, module_body = _extract_verilog_module_header(module_text, module_name)
        modules.append((module_name, module_body))
        cursor = block_end


def _collect_instantiated_modules(module_body: str, known_modules: frozenset[str]) -> tuple[str, ...]:
    cleaned = _strip_verilog_comments(module_body)
    discovered: list[str] = []
    pattern = re.compile(r"(?m)^\s*([A-Za-z_][\w$]*)\b")
    keywords = {
        "always",
        "assign",
        "case",
        "else",
        "end",
        "for",
        "function",
        "generate",
        "if",
        "initial",
        "input",
        "localparam",
        "logic",
        "output",
        "parameter",
        "reg",
        "task",
        "wire",
    }
    for match in pattern.finditer(cleaned):
        module_name = match.group(1)
        if module_name not in known_modules or module_name in keywords:
            continue
        cursor = match.end()
        while cursor < len(cleaned) and cleaned[cursor].isspace():
            cursor += 1
        if cursor < len(cleaned) and cleaned[cursor] == "#":
            cursor += 1
            while cursor < len(cleaned) and cleaned[cursor].isspace():
                cursor += 1
            if cursor >= len(cleaned) or cleaned[cursor] != "(":
                continue
            _ignored, cursor = _extract_balanced(cleaned, cursor)
            while cursor < len(cleaned) and cleaned[cursor].isspace():
                cursor += 1
        instance_match = re.match(r"([A-Za-z_][\w$]*)", cleaned[cursor:])
        if instance_match is None:
            continue
        cursor += instance_match.end()
        while cursor < len(cleaned) and cleaned[cursor].isspace():
            cursor += 1
        if cursor < len(cleaned) and cleaned[cursor] == "(" and module_name not in discovered:
            discovered.append(module_name)
    return tuple(discovered)


def _strip_verilog_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*?$", "", text, flags=re.MULTILINE)


def _extract_verilog_module_header(verilog_text: str, module_name: str) -> tuple[str | None, str, str]:
    pattern = re.compile(rf"\bmodule\s+{re.escape(module_name)}\b", re.MULTILINE)
    match = pattern.search(verilog_text)
    if match is None:
        raise ValueError(f"verilog source does not define module '{module_name}'")
    cursor = match.end()
    while cursor < len(verilog_text) and verilog_text[cursor].isspace():
        cursor += 1
    parameter_block: str | None = None
    if cursor < len(verilog_text) and verilog_text[cursor] == "#":
        cursor += 1
        while cursor < len(verilog_text) and verilog_text[cursor].isspace():
            cursor += 1
        if cursor >= len(verilog_text) or verilog_text[cursor] != "(":
            raise ValueError(f"module '{module_name}' uses unsupported parameter syntax")
        parameter_block, cursor = _extract_balanced(verilog_text, cursor)
        while cursor < len(verilog_text) and verilog_text[cursor].isspace():
            cursor += 1
    if cursor >= len(verilog_text) or verilog_text[cursor] != "(":
        raise ValueError(f"module '{module_name}' must use ANSI-style port declarations")
    port_block, cursor = _extract_balanced(verilog_text, cursor)
    body_start = cursor
    while body_start < len(verilog_text) and verilog_text[body_start].isspace():
        body_start += 1
    if body_start < len(verilog_text) and verilog_text[body_start] == ";":
        body_start += 1
    end_match = re.search(r"\bendmodule\b", verilog_text[body_start:], re.MULTILINE)
    if end_match is None:
        raise ValueError(f"verilog source module '{module_name}' is missing endmodule")
    module_body = verilog_text[body_start : body_start + end_match.start()]
    return parameter_block, port_block, module_body


def _extract_balanced(text: str, start_index: int) -> tuple[str, int]:
    if text[start_index] != "(":
        raise ValueError("balanced extraction expects '('")
    depth = 0
    current: list[str] = []
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
            if depth > 1:
                current.append(char)
            continue
        if char == ")":
            depth -= 1
            if depth == 0:
                return "".join(current), index + 1
            current.append(char)
            continue
        current.append(char)
    raise ValueError("unterminated parenthesized block in Verilog module header")


def _split_top_level_csv(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    for char in text:
        if char == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth > 0:
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth > 0:
            brace_depth -= 1
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_verilog_parameters(parameter_block: str | None) -> dict[str, dict[str, object]]:
    if parameter_block is None or not parameter_block.strip():
        return {}
    parameters: dict[str, dict[str, object]] = {}
    for entry in _split_top_level_csv(parameter_block):
        parameter = _parse_verilog_parameter(entry)
        parameters[parameter["name"]] = parameter["payload"]
    return parameters


def _parse_verilog_parameter(entry: str) -> dict[str, object]:
    cleaned = re.sub(r"\s+", " ", entry.strip())
    if cleaned.startswith("parameter "):
        cleaned = cleaned[len("parameter ") :]
    elif cleaned.startswith("localparam "):
        cleaned = cleaned[len("localparam ") :]
    else:
        raise ValueError(f"unsupported Verilog parameter declaration '{entry}'")
    name_part, sep, default_text = cleaned.partition("=")
    if sep != "=":
        raise ValueError(f"Verilog parameter '{entry}' must define a default value")
    left_tokens = name_part.strip().split()
    if not left_tokens:
        raise ValueError(f"Verilog parameter '{entry}' is missing a name")
    name = left_tokens[-1]
    default = default_text.strip()
    payload: dict[str, object] = {"kind": _infer_parameter_kind(default), "default": default}
    if "signed" in left_tokens:
        payload["signed"] = True
    return {"name": name, "payload": payload}


def _infer_parameter_kind(default_text: str) -> str:
    lowered = default_text.strip().lower()
    if lowered in {"true", "false"}:
        return "bool"
    if re.fullmatch(r"\d+", lowered):
        return "int"
    return "string"


def _parse_verilog_ports(port_block: str, *, module_body: str | None = None) -> dict[str, dict[str, str]]:
    entries = _split_top_level_csv(port_block)
    if entries and all(not re.match(r"^\s*(input|output|inout)\b", entry) for entry in entries):
        return _parse_nonansi_verilog_ports(entries, module_body or "")
    ports: dict[str, dict[str, str]] = {}
    for entry in entries:
        name, payload = _parse_verilog_port(entry)
        ports[name] = payload
    return ports


def _parse_nonansi_verilog_ports(header_entries: list[str], module_body: str) -> dict[str, dict[str, str]]:
    declared_ports: dict[str, dict[str, str]] = {}
    for declaration in _collect_nonansi_port_declarations(module_body):
        for name, payload in _parse_nonansi_port_declaration(declaration):
            declared_ports[name] = payload
    ports: dict[str, dict[str, str]] = {}
    for entry in header_entries:
        name = entry.strip()
        if not name:
            continue
        payload = declared_ports.get(name)
        if payload is None:
            raise ValueError(f"unsupported Verilog port declaration '{entry}'")
        ports[name] = payload
    return ports


def _collect_nonansi_port_declarations(module_body: str) -> list[str]:
    declarations: list[str] = []
    current: list[str] = []
    depth = 0
    for char in module_body:
        current.append(char)
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth > 0:
            depth -= 1
        elif char == ";" and depth == 0:
            declaration = "".join(current).strip()
            current = []
            if re.match(r"^(input|output|inout)\b", re.sub(r"\s+", " ", declaration)):
                declarations.append(declaration[:-1].strip())
    return declarations


def _parse_nonansi_port_declaration(declaration: str) -> list[tuple[str, dict[str, str]]]:
    cleaned = re.sub(r"\s+", " ", declaration.strip())
    match = re.match(r"^(input|output|inout)\b\s*(.*)$", cleaned)
    if match is None:
        raise ValueError(f"unsupported Verilog port declaration '{declaration}'")
    direction = match.group(1)
    remainder = match.group(2).strip()
    signed = False
    if remainder.startswith(("wire ", "logic ", "reg ")):
        remainder = remainder.split(" ", 1)[1].strip()
    if remainder.startswith("signed "):
        signed = True
        remainder = remainder[len("signed ") :].strip()
    width_expr: str | None = None
    range_match = re.match(r"^\[(.+?)\]\s+(.*)$", remainder)
    if range_match is not None:
        width_expr = range_match.group(1).strip()
        remainder = range_match.group(2).strip()
    names = [name.strip() for name in remainder.split(",") if name.strip()]
    return [(name, {"dir": direction, "type": _verilog_port_type(width_expr, signed)}) for name in names]


def _parse_verilog_port(entry: str) -> tuple[str, dict[str, str]]:
    cleaned = re.sub(r"\s+", " ", entry.strip())
    match = re.match(r"^(input|output|inout)\b\s*(.*)$", cleaned)
    if match is None:
        raise ValueError(f"unsupported Verilog port declaration '{entry}'")
    direction = match.group(1)
    remainder = match.group(2).strip()
    signed = False
    if remainder.startswith(("wire ", "logic ", "reg ")):
        remainder = remainder.split(" ", 1)[1].strip()
    if remainder.startswith("signed "):
        signed = True
        remainder = remainder[len("signed ") :].strip()
    range_match = re.match(r"^\[(.+?)\]\s+(.*)$", remainder)
    width_expr: str | None = None
    if range_match is not None:
        width_expr = range_match.group(1).strip()
        remainder = range_match.group(2).strip()
    name = remainder.split()[-1]
    return name, {"dir": direction, "type": _verilog_port_type(width_expr, signed)}


def _verilog_port_type(width_expr: str | None, signed: bool) -> str:
    if width_expr is None:
        return "i1"
    match = re.fullmatch(r"(\d+)\s*:\s*0", width_expr)
    if match is None:
        return "TODO"
    width = int(match.group(1)) + 1
    return f"{'i' if signed else 'u'}{width}"
