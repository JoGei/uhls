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
) -> dict[str, object]:
    """Build one component-library stub from one Verilog module interface."""
    parameter_block, port_block = _extract_verilog_module_header(verilog_text, module_name)
    parameters = _parse_verilog_parameters(parameter_block)
    ports = _parse_verilog_ports(port_block)
    return {
        "kind": "combinational",
        "parameters": parameters,
        "hdl": {
            "language": "verilog",
            "module": module_name,
            "source": None if source_path is None else str(source_path),
        },
        "ports": ports,
        "supports": {op: _support_stub() for op in ops},
    }


def _support_stub() -> dict[str, object]:
    return {
        "ii": "TODO",
        "d": "TODO",
        "bind": "TODO",
    }


def _extract_verilog_module_header(verilog_text: str, module_name: str) -> tuple[str | None, str]:
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
    port_block, _ = _extract_balanced(verilog_text, cursor)
    return parameter_block, port_block


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


def _parse_verilog_ports(port_block: str) -> dict[str, dict[str, str]]:
    ports: dict[str, dict[str, str]] = {}
    for entry in _split_top_level_csv(port_block):
        name, payload = _parse_verilog_port(entry)
        ports[name] = payload
    return ports


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
