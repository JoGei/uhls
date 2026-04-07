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
        normalized[str(component_name)] = component_payload
    return normalized


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
