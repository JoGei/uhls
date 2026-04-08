"""Shared helpers for parameterized component-library specifications."""

from __future__ import annotations

import re
from typing import Any


def parse_component_spec(component_name: str) -> tuple[str, dict[str, str]]:
    """Split one ``NAME<k=v,...>`` component spec into base name and parameters."""
    match = re.fullmatch(r"([A-Za-z_][\w$]*)(?:<(.*)>)?", component_name)
    if match is None:
        raise ValueError(f"invalid component spec '{component_name}'")
    base_name, params_text = match.groups()
    if params_text is None or not params_text.strip():
        return base_name, {}
    params: dict[str, str] = {}
    for part in _split_component_params(params_text):
        key, sep, value = part.partition("=")
        if sep != "=" or not key or not value:
            raise ValueError(f"invalid component parameter '{part}' in '{component_name}'")
        params[key.strip()] = value.strip()
    return base_name, params


def format_component_spec(base_name: str, params: dict[str, str]) -> str:
    """Render one component spec in canonical textual form."""
    if not params:
        return base_name
    ordered_keys = [key for key in ("word_t", "word_len") if key in params]
    ordered_keys.extend(sorted(key for key in params if key not in {"word_t", "word_len"}))
    parts = [f"{key}={params[key]}" for key in ordered_keys]
    return f"{base_name}<{','.join(parts)}>"


def resolve_component_definition(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Resolve one component spec against one validated component library."""
    base_name, params = parse_component_spec(component_name)
    component = component_library.get(base_name)
    if component is None:
        raise ValueError(f"component library does not define component '{base_name}'")
    if not isinstance(component, dict):
        raise ValueError(f"component '{base_name}' must be an object")
    _validate_component_spec_params(base_name, params, component)
    return base_name, params, component


def validate_component_library(components: dict[str, object]) -> dict[str, dict[str, object]]:
    """Validate and normalize one shared component-library payload."""
    normalized: dict[str, dict[str, object]] = {}
    for component_name, component_payload in components.items():
        if not isinstance(component_payload, dict):
            raise ValueError(f"component '{component_name}' must be a JSON object")
        kind = component_payload.get("kind")
        if kind is not None and kind not in {"combinational", "sequential", "pipelined", "memory", "fifo", "stream"}:
            raise ValueError(
                f"component '{component_name}' kind must be one of: combinational, sequential, pipelined, memory, fifo, stream"
            )
        hdl = component_payload.get("hdl")
        if hdl is not None:
            if not isinstance(hdl, dict):
                raise ValueError(f"component '{component_name}' must define object-valued 'hdl'")
            language = hdl.get("language")
            if language not in {"verilog", "vhdl", "systemc"}:
                raise ValueError(
                    f"component '{component_name}' hdl.language must be one of: verilog, vhdl, systemc"
                )
            module_name = hdl.get("module")
            if not isinstance(module_name, str) or not module_name.strip():
                raise ValueError(f"component '{component_name}' hdl.module must be one non-empty string")
            source = hdl.get("source")
            if source is not None and (not isinstance(source, str) or not source.strip()):
                raise ValueError(f"component '{component_name}' hdl.source must be one non-empty string")
        parameters = component_payload.get("parameters")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError(f"component '{component_name}' must define object-valued 'parameters'")
            for parameter_name, parameter_payload in parameters.items():
                if not isinstance(parameter_payload, dict):
                    raise ValueError(
                        f"component '{component_name}' parameter '{parameter_name}' must be an object"
                    )
                kind = parameter_payload.get("kind")
                if kind not in {"type", "int", "bool", "string"}:
                    raise ValueError(
                        f"component '{component_name}' parameter '{parameter_name}' must use kind=type|int|bool|string"
                    )
                required = parameter_payload.get("required")
                if required is not None and not isinstance(required, bool):
                    raise ValueError(
                        f"component '{component_name}' parameter '{parameter_name}' must use boolean 'required'"
                    )
        supports = component_payload.get("supports")
        if supports is not None:
            if not isinstance(supports, dict):
                raise ValueError(f"component '{component_name}' must define object-valued 'supports'")
            for operation_name, support_payload in supports.items():
                if not isinstance(support_payload, dict):
                    raise ValueError(
                        f"component '{component_name}' support '{operation_name}' must be a JSON object"
                    )
                _validate_component_support(component_name, str(kind) if kind is not None else None, operation_name, support_payload)
        ports = component_payload.get("ports")
        if ports is not None:
            if not isinstance(ports, dict):
                raise ValueError(f"component '{component_name}' must define object-valued 'ports'")
            _validate_component_ports(component_name, ports)
        normalized[str(component_name)] = component_payload
    return normalized


def _validate_component_support(
    component_name: str,
    kind: str | None,
    operation_name: str,
    support_payload: dict[str, Any],
) -> None:
    ii = support_payload.get("ii")
    delay = support_payload.get("d")
    if not isinstance(ii, int) or not isinstance(delay, int):
        return
    if ii < 1 or delay < 1:
        raise ValueError(
            f"component '{component_name}' support '{operation_name}' must define positive integer ii/d"
        )
    if kind == "sequential" and ii < delay:
        raise ValueError(
            f"component '{component_name}' is kind=sequential but support '{operation_name}' uses ii={ii} < d={delay}"
        )


def _validate_component_ports(component_name: str, ports: dict[str, object]) -> None:
    seen_clock = 0
    seen_reset = 0
    for port_name, port_payload in ports.items():
        if not isinstance(port_payload, dict):
            raise ValueError(f"component '{component_name}' port '{port_name}' must be a JSON object")
        direction = port_payload.get("dir")
        if direction not in {"input", "output"}:
            raise ValueError(f"component '{component_name}' port '{port_name}' dir must be 'input' or 'output'")
        type_name = port_payload.get("type")
        if not isinstance(type_name, str) or not type_name:
            raise ValueError(f"component '{component_name}' port '{port_name}' must define string 'type'")
        if type_name == "clock":
            if direction != "input":
                raise ValueError(f"component '{component_name}' clock port '{port_name}' must be an input")
            seen_clock += 1
        elif type_name == "reset":
            if direction != "input":
                raise ValueError(f"component '{component_name}' reset port '{port_name}' must be an input")
            seen_reset += 1
            reset_kind = port_payload.get("kind")
            reset_active = port_payload.get("active")
            if reset_kind not in {"sync", "async"}:
                raise ValueError(
                    f"component '{component_name}' reset port '{port_name}' must define kind=sync|async"
                )
            if reset_active not in {"hi", "lo"}:
                raise ValueError(
                    f"component '{component_name}' reset port '{port_name}' must define active=hi|lo"
                )
        else:
            if "kind" in port_payload or "active" in port_payload:
                raise ValueError(
                    f"component '{component_name}' non-reset port '{port_name}' must not define reset kind/active"
                )
    if seen_clock > 1:
        raise ValueError(f"component '{component_name}' may define at most one clock port")
    if seen_reset > 1:
        raise ValueError(f"component '{component_name}' may define at most one reset port")


def _validate_component_spec_params(base_name: str, params: dict[str, str], component: dict[str, Any]) -> None:
    parameter_specs = component.get("parameters")
    if parameter_specs is None:
        if params:
            raise ValueError(
                f"component '{base_name}' does not declare parameters but spec provided {', '.join(sorted(params))}"
            )
        return
    if not isinstance(parameter_specs, dict):
        raise ValueError(f"component '{base_name}' must define object-valued 'parameters'")
    declared_names = set(parameter_specs)
    unknown = sorted(name for name in params if name not in declared_names)
    if unknown:
        raise ValueError(
            f"component '{base_name}' does not declare parameter(s): {', '.join(unknown)}"
        )
    for parameter_name, parameter_payload in parameter_specs.items():
        if not isinstance(parameter_payload, dict):
            raise ValueError(f"component '{base_name}' parameter '{parameter_name}' must be an object")
        required = bool(parameter_payload.get("required", False))
        if required and parameter_name not in params:
            raise ValueError(f"component '{base_name}' spec must provide required parameter '{parameter_name}'")
        if parameter_name in params:
            _validate_component_param_value(base_name, parameter_name, params[parameter_name], parameter_payload)


def _validate_component_param_value(
    component_name: str,
    parameter_name: str,
    value: str,
    parameter_payload: dict[str, Any],
) -> None:
    kind = parameter_payload.get("kind")
    if kind == "int":
        if not re.fullmatch(r"\d+", value):
            raise ValueError(
                f"component '{component_name}' parameter '{parameter_name}' must be one non-negative integer"
            )
        return
    if kind == "bool":
        if value not in {"true", "false"}:
            raise ValueError(
                f"component '{component_name}' parameter '{parameter_name}' must be 'true' or 'false'"
            )
        return
    if kind == "type":
        if not re.fullmatch(r"[A-Za-z_][\w$]*(?:<[^>\n]+>)?", value):
            raise ValueError(
                f"component '{component_name}' parameter '{parameter_name}' must be one textual type name"
            )
        return
    if kind == "string":
        if not value:
            raise ValueError(
                f"component '{component_name}' parameter '{parameter_name}' must be one non-empty string"
            )
        return
    raise ValueError(
        f"component '{component_name}' parameter '{parameter_name}' must use kind=type|int|bool|string"
    )


def _split_component_params(params_text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in params_text:
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if char == "<":
            depth += 1
        elif char == ">" and depth > 0:
            depth -= 1
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts
