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


def resolve_component_type(type_name: str, params: dict[str, str]) -> str:
    """Resolve one semantic component type against one parameter environment."""
    return params.get(type_name, type_name)


def materialize_hdl_component_spec(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
) -> str:
    """Resolve one semantic component spec into one concrete HDL instance spec."""
    base_name, params, component = resolve_component_definition(component_library, component_name)
    hdl = component.get("hdl")
    if not isinstance(hdl, dict):
        return component_name
    module_name = hdl.get("module")
    if not isinstance(module_name, str) or not module_name.strip():
        return component_name
    hdl_parameters = hdl.get("parameters")
    if hdl_parameters is None:
        return module_name
    if not isinstance(hdl_parameters, dict):
        raise ValueError(f"component '{base_name}' hdl.parameters must be an object")
    resolved: dict[str, str] = {}
    for parameter_name, parameter_expr in hdl_parameters.items():
        if not isinstance(parameter_expr, str) or not parameter_expr.strip():
            raise ValueError(
                f"component '{base_name}' hdl.parameters '{parameter_name}' must be one non-empty string"
            )
        resolved[str(parameter_name)] = _resolve_hdl_parameter_expr(base_name, parameter_expr.strip(), params)
    return format_component_spec(module_name, resolved)


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
            sources = hdl.get("sources")
            if source is not None and (not isinstance(source, str) or not source.strip()):
                raise ValueError(f"component '{component_name}' hdl.source must be one non-empty string")
            if sources is not None:
                if not isinstance(sources, list) or not sources:
                    raise ValueError(f"component '{component_name}' hdl.sources must be one non-empty list")
                for index, path in enumerate(sources):
                    if not isinstance(path, str) or not path.strip():
                        raise ValueError(
                            f"component '{component_name}' hdl.sources[{index}] must be one non-empty string"
                        )
            if source is None and sources is None:
                pass
            include_dirs = hdl.get("include_dirs")
            if include_dirs is not None:
                if not isinstance(include_dirs, list) or not include_dirs:
                    raise ValueError(f"component '{component_name}' hdl.include_dirs must be one non-empty list")
                for index, path in enumerate(include_dirs):
                    if not isinstance(path, str) or not path.strip():
                        raise ValueError(
                            f"component '{component_name}' hdl.include_dirs[{index}] must be one non-empty string"
                        )
            defines = hdl.get("defines")
            if defines is not None:
                if not isinstance(defines, list) or not defines:
                    raise ValueError(f"component '{component_name}' hdl.defines must be one non-empty list")
                for index, define in enumerate(defines):
                    if not isinstance(define, str) or not define.strip():
                        raise ValueError(
                            f"component '{component_name}' hdl.defines[{index}] must be one non-empty string"
                        )
            hdl_parameters = hdl.get("parameters")
            if hdl_parameters is not None:
                if not isinstance(hdl_parameters, dict):
                    raise ValueError(f"component '{component_name}' hdl.parameters must be an object")
                for parameter_name, parameter_expr in hdl_parameters.items():
                    if not isinstance(parameter_expr, str) or not parameter_expr.strip():
                        raise ValueError(
                            f"component '{component_name}' hdl.parameters '{parameter_name}' must be one non-empty string"
                        )
        parameters = component_payload.get("parameters")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError(f"component '{component_name}' must define object-valued 'parameters'")
            for parameter_name, parameter_payload in parameters.items():
                if not isinstance(parameter_payload, dict):
                    raise ValueError(
                        f"component '{component_name}' parameter '{parameter_name}' must be an object"
                    )
                parameter_kind = parameter_payload.get("kind")
                if parameter_kind not in {"type", "int", "bool", "string"}:
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
                support_types = support_payload.get("types")
                if support_types is not None:
                    if not isinstance(support_types, dict):
                        raise ValueError(
                            f"component '{component_name}' support '{operation_name}' must use object-valued 'types'"
                        )
                    for binding_name, binding_type in support_types.items():
                        if not isinstance(binding_type, str) or not binding_type:
                            raise ValueError(
                                f"component '{component_name}' support '{operation_name}' types '{binding_name}' must be a non-empty string"
                            )
                _validate_component_support(component_name, str(kind) if kind is not None else None, operation_name, support_payload)
        ports = component_payload.get("ports")
        if ports is not None:
            if not isinstance(ports, dict):
                raise ValueError(f"component '{component_name}' must define object-valued 'ports'")
            _validate_component_ports(component_name, ports)
        memory_shape = component_payload.get("memory")
        if memory_shape is not None:
            if kind != "memory":
                raise ValueError(f"component '{component_name}' may define 'memory' only when kind=memory")
            if not isinstance(memory_shape, dict):
                raise ValueError(f"component '{component_name}' must define object-valued 'memory'")
            word_t = memory_shape.get("word_t")
            word_len = memory_shape.get("word_len")
            if not isinstance(word_t, str) or not word_t:
                raise ValueError(f"component '{component_name}' memory.word_t must be one non-empty string")
            if not isinstance(word_len, int) or word_len <= 0:
                raise ValueError(f"component '{component_name}' memory.word_len must be one positive integer")
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
        tie_value = port_payload.get("tie")
        if tie_value is not None:
            if direction != "input":
                raise ValueError(f"component '{component_name}' tied port '{port_name}' must be an input")
            if type_name in {"clock", "reset"}:
                raise ValueError(
                    f"component '{component_name}' semantic port '{port_name}' must not define tie"
                )
            if not isinstance(tie_value, str) or not tie_value:
                raise ValueError(
                    f"component '{component_name}' tied port '{port_name}' must define non-empty string tie"
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


def _resolve_hdl_parameter_expr(component_name: str, expr: str, params: dict[str, str]) -> str:
    bits_match = re.fullmatch(r"\$bits\(([^()\s]+)\)", expr)
    if bits_match is not None:
        semantic_name = bits_match.group(1)
        semantic_type = params.get(semantic_name)
        if semantic_type is None:
            raise ValueError(
                f"component '{component_name}' hdl parameter expression '{expr}' references unknown semantic parameter '{semantic_name}'"
            )
        return str(_uir_type_bits(component_name, semantic_name, semantic_type))
    if expr in params:
        return params[expr]
    return expr


def _uir_type_bits(component_name: str, parameter_name: str, type_name: str) -> int:
    match = re.fullmatch(r"[iu](\d+)", type_name)
    if match is None:
        raise ValueError(
            f"component '{component_name}' hdl parameter '{parameter_name}' uses unsupported $bits type '{type_name}'"
        )
    return int(match.group(1))


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
