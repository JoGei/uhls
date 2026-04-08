"""Shared memory-component helpers for wrapped protocol builders."""

from __future__ import annotations

from typing import Any

from uhls.backend.hls.lib import (
    materialize_hdl_component_spec,
    resolve_component_definition,
    resolve_component_type,
)

_NON_LITERAL_BINDINGS = frozenset({"operand0", "operand1", "operand2", "result", "opcode"})


def materialized_memory_instance_spec(
    component_library: dict[str, dict[str, Any]],
    component_spec: str,
) -> str:
    """Resolve one semantic memory component spec into one HDL instance spec."""
    return materialize_hdl_component_spec(component_library, component_spec)


def resolved_component_ports(
    component_library: dict[str, dict[str, Any]],
    component_spec: str,
) -> dict[str, dict[str, Any]]:
    """Return resolved port payloads keyed by port name."""
    _base_name, params, component = resolve_component_definition(component_library, component_spec)
    ports = component.get("ports")
    if not isinstance(ports, dict):
        raise ValueError(f"component '{component_spec}' must define object-valued 'ports'")
    resolved: dict[str, dict[str, Any]] = {}
    for port_name, port_payload in ports.items():
        if not isinstance(port_payload, dict):
            raise ValueError(f"component '{component_spec}' port '{port_name}' must be an object")
        payload = dict(port_payload)
        port_type = payload.get("type")
        if isinstance(port_type, str) and port_type:
            payload["type"] = resolve_component_type(port_type, params)
        resolved[str(port_name)] = payload
    return resolved


def component_tied_input_ports(
    component_library: dict[str, dict[str, Any]],
    component_spec: str,
) -> dict[str, str]:
    """Return tied input ports for one memory component."""
    ties: dict[str, str] = {}
    for port_name, payload in resolved_component_ports(component_library, component_spec).items():
        if payload.get("dir") != "input":
            continue
        tie_expr = payload.get("tie")
        if isinstance(tie_expr, str) and tie_expr:
            ties[port_name] = tie_expr
    return ties


def component_semantic_port_signal(
    component_library: dict[str, dict[str, Any]],
    component_spec: str,
    port_name: str,
) -> str | None:
    """Return one pre-wired signal name for one semantic port, if any."""
    payload = resolved_component_ports(component_library, component_spec).get(port_name)
    if payload is None:
        return None
    port_type = payload.get("type")
    if port_type == "clock":
        return "clk"
    if port_type == "reset":
        active = payload.get("active")
        return "rst" if active == "hi" else "rst_n"
    return None


def component_memory_port_name(
    component_library: dict[str, dict[str, Any]],
    component_spec: str,
    role: str,
) -> str | None:
    """Resolve one semantic memory role onto one concrete component port."""
    ports = resolved_component_ports(component_library, component_spec)
    if role in ports:
        return role
    semantic_binding = {
        "addr": "operand1",
        "wdata": "operand2",
        "rdata": "result",
    }.get(role)
    if semantic_binding is not None:
        for opcode_name in ("load", "store"):
            port_name = component_support_port_for_binding(
                component_library,
                component_spec,
                opcode_name,
                semantic_binding,
            )
            if port_name is not None:
                direction = ports[port_name].get("dir")
                if role == "rdata" and direction == "output":
                    return port_name
                if role != "rdata" and direction == "input":
                    return port_name
    if role == "we":
        for literal in ("true", "false"):
            port_name = component_support_port_for_binding(
                component_library,
                component_spec,
                "store",
                literal,
            )
            if port_name is not None and ports[port_name].get("dir") == "input":
                return port_name
    return None


def component_memory_port_type(
    component_library: dict[str, dict[str, Any]],
    component_spec: str,
    role: str,
) -> str | None:
    """Resolve the type of one semantic memory role."""
    port_name = component_memory_port_name(component_library, component_spec, role)
    if port_name is None:
        return None
    payload = resolved_component_ports(component_library, component_spec).get(port_name)
    port_type = None if payload is None else payload.get("type")
    return port_type if isinstance(port_type, str) and port_type else None


def component_support_port_for_binding(
    component_library: dict[str, dict[str, Any]],
    component_spec: str,
    opcode_name: str,
    binding_value: str,
) -> str | None:
    """Return one support-mapped port for one binding value."""
    _base_name, _params, component = resolve_component_definition(component_library, component_spec)
    supports = component.get("supports")
    if not isinstance(supports, dict):
        raise ValueError(f"component '{component_spec}' must define object-valued 'supports'")
    support = supports.get(opcode_name)
    if support is None:
        return None
    if not isinstance(support, dict):
        raise ValueError(f"component '{component_spec}' support '{opcode_name}' must be an object")
    binding = support.get("bind")
    if not isinstance(binding, dict):
        return None
    for port_name, bound_value in binding.items():
        if bound_value == binding_value:
            return str(port_name)
    return None


def component_memory_control_literals(
    component_library: dict[str, dict[str, Any]],
    component_spec: str,
) -> dict[str, tuple[str | None, str | None]]:
    """Return literal load/store control expressions keyed by input port."""
    ports = resolved_component_ports(component_library, component_spec)
    _base_name, _params, component = resolve_component_definition(component_library, component_spec)
    supports = component.get("supports")
    if not isinstance(supports, dict):
        raise ValueError(f"component '{component_spec}' must define object-valued 'supports'")
    load_bind = supports.get("load", {})
    store_bind = supports.get("store", {})
    if not isinstance(load_bind, dict):
        load_bind = {}
    if not isinstance(store_bind, dict):
        store_bind = {}
    load_map = load_bind.get("bind", {})
    store_map = store_bind.get("bind", {})
    if not isinstance(load_map, dict):
        load_map = {}
    if not isinstance(store_map, dict):
        store_map = {}
    control_ports = {
        str(port_name)
        for port_name, payload in ports.items()
        if payload.get("dir") == "input"
        and str(port_name) not in component_tied_input_ports(component_library, component_spec)
        and payload.get("type") not in {"clock", "reset"}
        and str(port_name) not in {
            component_memory_port_name(component_library, component_spec, "addr"),
            component_memory_port_name(component_library, component_spec, "wdata"),
        }
    }
    literals: dict[str, tuple[str | None, str | None]] = {}
    for port_name in sorted(control_ports):
        load_expr = _literal_binding(load_map.get(port_name))
        store_expr = _literal_binding(store_map.get(port_name))
        if load_expr is None and store_expr is None:
            continue
        literals[port_name] = (load_expr, store_expr)
    return literals


def _literal_binding(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value in _NON_LITERAL_BINDINGS:
        return None
    return value

