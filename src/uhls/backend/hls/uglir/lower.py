"""FSM-to-uglir lowering entrypoint."""

from __future__ import annotations

from math import ceil, log2
from typing import Any

from uhls.backend.uhir.model import (
    UHIRAssign,
    UHIRAttach,
    UHIRDesign,
    UHIRGlueMux,
    UHIRGlueMuxCase,
    UHIRPort,
    UHIRResource,
    UHIRSeqBlock,
    UHIRSeqUpdate,
)


def lower_fsm_to_uglir(design: UHIRDesign, component_library: dict[str, dict[str, Any]] | None = None) -> UHIRDesign:
    """Lower one static fsm-stage µhIR design to one initial uglir shell."""
    if design.stage != "fsm":
        raise ValueError(f"uglir lowering expects fsm-stage µhIR input, got stage '{design.stage}'")
    if any(controller.attributes.get("protocol") != "req_resp" for controller in design.controllers):
        raise ValueError("initial uglir lowering currently supports only one static top-level req_resp controller")
    if len(design.controllers) != 1:
        raise ValueError("initial uglir lowering currently expects exactly one controller")

    controller = design.controllers[0]
    lowered = UHIRDesign(name=design.name, stage="uglir")
    memory_interfaces = _memory_interfaces(design, component_library)
    _validate_memory_interface_schedule(controller, memory_interfaces)
    lowered.inputs = [
        UHIRPort("input", "clk", "clock"),
        UHIRPort("input", "rst", "i1"),
        UHIRPort("input", "req_valid", "i1"),
        UHIRPort("input", "resp_ready", "i1"),
    ]
    lowered.inputs.extend(_lowered_data_inputs(design, memory_interfaces))
    lowered.outputs = [
        UHIRPort("output", "req_ready", "i1"),
        UHIRPort("output", "resp_valid", "i1"),
    ]
    lowered.outputs.extend(_lowered_data_outputs(design, memory_interfaces))
    lowered.constants = list(design.constants)

    state_type = _state_type(controller)
    lowered.resources.append(UHIRResource("reg", "state", state_type))
    lowered.resources.append(UHIRResource("net", "next_state", state_type))
    lowered.resources.append(UHIRResource("net", "req_fire", "i1"))
    lowered.resources.append(UHIRResource("net", "resp_fire", "i1"))
    for interface in memory_interfaces.values():
        lowered.resources.append(UHIRResource("port", interface["memory_name"], interface["component_name"], interface["memory_name"]))

    for resource in design.resources:
        if resource.kind == "fu":
            lowered.resources.append(UHIRResource("inst", resource.id, resource.value))
            if component_library is None:
                lowered.resources.append(UHIRResource("net", f"{resource.id}_go", "i1"))
                result_type = _instance_result_type(design, resource.id)
                if result_type is not None:
                    lowered.resources.append(UHIRResource("net", f"{resource.id}_y", result_type))
            else:
                for port_name, port_type in _instance_ports(component_library, resource.value):
                    lowered.resources.append(UHIRResource("net", f"{resource.id}_{port_name}", port_type))
        elif resource.kind == "reg":
            lowered.resources.append(UHIRResource("reg", resource.id, resource.value))

    state_code = {state.name: state.attributes["code"] for state in controller.states}
    latch_targets = sorted(
        {
            register
            for emit in controller.emits
            for register in emit.attributes.get("latch", ())
        }
    )
    for register in latch_targets:
        lowered.resources.append(UHIRResource("net", f"latch_{register}", "i1"))
        lowered.resources.append(UHIRResource("net", f"sel_{register}", "ctrl"))
        lowered.resources.append(UHIRResource("mux", f"mx_{register}", _resource_value(design, register)))

    lowered.assigns.extend(
        [
            UHIRAssign("req_fire", "req_valid & req_ready"),
            UHIRAssign("resp_fire", "resp_valid & resp_ready"),
            UHIRAssign("req_ready", _state_eq_expr(state_code["IDLE"])),
            UHIRAssign("resp_valid", _state_eq_expr(state_code["DONE"])),
            UHIRAssign("next_state", _next_state_expr(controller, state_code)),
        ]
    )

    for resource in design.resources:
        if resource.kind != "fu":
            continue
        if component_library is None:
            lowered.assigns.append(UHIRAssign(f"{resource.id}_go", _issue_expr(controller, resource.id, state_code)))
            lowered.attachments.append(UHIRAttach(resource.id, "go", f"{resource.id}_go"))
            if any(candidate.id == f"{resource.id}_y" for candidate in lowered.resources):
                lowered.attachments.append(UHIRAttach(resource.id, "y", f"{resource.id}_y"))
            continue

        port_names = _instance_port_names(component_library, resource.value)
        if "go" in port_names:
            lowered.assigns.append(UHIRAssign(f"{resource.id}_go", _issue_expr(controller, resource.id, state_code)))
        if "op" in port_names:
            lowered.assigns.append(
                UHIRAssign(
                    f"{resource.id}_op",
                    _opcode_expr(design, controller, resource.id, resource.value, state_code, component_library),
                )
            )
        for port_name, port_type in _instance_input_ports(component_library, resource.value):
            if port_name in {"go", "op"}:
                continue
            lowered.assigns.append(
                UHIRAssign(
                    f"{resource.id}_{port_name}",
                    _operand_port_expr(
                        design,
                        controller,
                        resource.id,
                        resource.value,
                        port_name,
                        port_type,
                        state_code,
                        component_library,
                    ),
                )
            )
        for port_name in port_names:
            lowered.attachments.append(UHIRAttach(resource.id, port_name, f"{resource.id}_{port_name}"))

    for register in latch_targets:
        lowered.assigns.append(UHIRAssign(f"latch_{register}", _latch_expr(controller, register, state_code)))
        lowered.assigns.append(
            UHIRAssign(
                f"sel_{register}",
                _select_expr(design, controller, register, state_code, component_library),
            )
        )
        lowered.glue_muxes.append(_build_register_mux(design, register, component_library))

    for assign in _memory_interface_assigns(memory_interfaces):
        lowered.assigns.append(assign)

    for output_name, driver in _output_drivers(design, component_library).items():
        lowered.assigns.append(UHIRAssign(output_name, driver))

    seq_block = UHIRSeqBlock(
        clock="clk",
        reset="rst",
        reset_updates=[UHIRSeqUpdate("state", str(state_code["IDLE"]))],
        updates=[UHIRSeqUpdate("state", "next_state")],
    )
    for register in latch_targets:
        seq_block.updates.append(UHIRSeqUpdate(register, f"mx_{register}", f"latch_{register}"))
    lowered.seq_blocks.append(seq_block)
    return lowered


def _state_type(controller) -> str:
    encoding = controller.attributes.get("encoding")
    count = max(len(controller.states), 1)
    if encoding == "one_hot":
        return f"u{count}"
    width = max(1, ceil(log2(count)))
    return f"u{width}"


def _state_eq_expr(code: int) -> str:
    return f"state == {code}"


def _next_state_expr(controller, state_code: dict[str, int]) -> str:
    branches: list[str] = []
    for transition in controller.transitions:
        condition = transition.attributes.get("when")
        transition_condition = "true" if not isinstance(condition, str) or not condition else condition
        branches.append(
            f"({ _state_eq_expr(state_code[transition.source]) } && ({transition_condition})) ? {state_code[transition.target]}"
        )
    return " : ".join(branches) + f" : {state_code['IDLE']}"


def _issue_expr(controller, resource_id: str, state_code: dict[str, int]) -> str:
    active_states = [
        state_name
        for state_name, attrs in ((emit.state, emit.attributes) for emit in controller.emits)
        for issue in _iter_issue_actions(attrs)
        if issue.split("<-", 1)[0] == resource_id
    ]
    if not active_states:
        return "false"
    return " | ".join(_state_eq_expr(state_code[state_name]) for state_name in active_states)


def _latch_expr(controller, register: str, state_code: dict[str, int]) -> str:
    active_states = [
        emit.state
        for emit in controller.emits
        if register in emit.attributes.get("latch", ())
    ]
    if not active_states:
        return "false"
    return " | ".join(_state_eq_expr(state_code[state_name]) for state_name in active_states)


def _select_expr(
    design: UHIRDesign,
    controller,
    register: str,
    state_code: dict[str, int],
    component_library: dict[str, dict[str, Any]] | None = None,
) -> str:
    choices = _register_state_sources(design, controller, register, component_library)
    if not choices:
        return "hold"
    branches = [f"{_state_eq_expr(state_code[state])} ? {source}" for state, source in sorted(choices.items(), key=lambda item: state_code[item[0]])]
    return " : ".join(branches) + " : hold"


def _build_register_mux(
    design: UHIRDesign,
    register: str,
    component_library: dict[str, dict[str, Any]] | None,
) -> UHIRGlueMux:
    register_type = _resource_value(design, register)
    glue_mux = UHIRGlueMux(name=f"mx_{register}", type=register_type, select=f"sel_{register}")
    glue_mux.cases.append(UHIRGlueMuxCase("hold", register))
    seen_sources = {register}
    for source in _register_possible_sources(design, register, component_library):
        if source in seen_sources:
            continue
        glue_mux.cases.append(UHIRGlueMuxCase(source, source))
        seen_sources.add(source)
    return glue_mux


def _register_possible_sources(
    design: UHIRDesign,
    register: str,
    component_library: dict[str, dict[str, Any]] | None,
) -> list[str]:
    producer_to_instance = _producer_instance_map(design)
    sources: list[str] = []
    for region in design.regions:
        for binding in region.value_bindings:
            if binding.register != register:
                continue
            instance = producer_to_instance.get(binding.producer)
            if instance is not None:
                sources.append(_instance_result_signal(design, instance, component_library))
    return sources


def _register_state_sources(
    design: UHIRDesign,
    controller,
    register: str,
    component_library: dict[str, dict[str, Any]] | None,
) -> dict[str, str]:
    producer_to_instance = _producer_instance_map(design)
    live_start_to_instance: dict[int, str] = {}
    for region in design.regions:
        for binding in region.value_bindings:
            if binding.register != register:
                continue
            instance = producer_to_instance.get(binding.producer)
            if instance is None:
                continue
            live_start_to_instance[binding.live_start] = _instance_result_signal(design, instance, component_library)
    choices: dict[str, str] = {}
    for emit in controller.emits:
        if register not in emit.attributes.get("latch", ()):
            continue
        if emit.state.startswith("T"):
            time_step = int(emit.state[1:])
            source = live_start_to_instance.get(time_step)
            if source is not None:
                choices[emit.state] = source
    return choices


def _producer_instance_map(design: UHIRDesign) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for region in design.regions:
        local_names: dict[str, str] = {}
        for node in region.nodes:
            bind = node.attributes.get("bind")
            if isinstance(bind, str):
                local_names[node.id] = bind
        for source_map in region.mappings:
            bind = local_names.get(source_map.node_id)
            if bind is not None:
                mapping[source_map.source_id] = bind
        mapping.update(local_names)
    return mapping


def _producer_register_map(design: UHIRDesign) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for region in design.regions:
        for binding in region.value_bindings:
            mapping.setdefault(binding.producer, binding.register)
    return mapping


def _resource_value(design: UHIRDesign, resource_id: str) -> str:
    for resource in design.resources:
        if resource.id == resource_id:
            return resource.value
    raise ValueError(f"unknown resource '{resource_id}'")


def _instance_result_type(design: UHIRDesign, resource_id: str) -> str | None:
    for region in design.regions:
        for node in region.nodes:
            if node.attributes.get("bind") == resource_id and node.result_type is not None:
                return node.result_type
    return None


def _is_memref_type(type_name: str) -> bool:
    return type_name.startswith("memref<") and type_name.endswith(">")


def _component_kind(component_library: dict[str, dict[str, Any]], component_name: str) -> str | None:
    component = component_library.get(component_name)
    if component is None:
        raise ValueError(f"component library does not define component '{component_name}'")
    kind = component.get("kind")
    if kind is None:
        return None
    if not isinstance(kind, str):
        raise ValueError(f"component '{component_name}' must define string 'kind'")
    return kind


def _memory_port_type(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
    port_name: str,
) -> str | None:
    for candidate_name, candidate_type in _instance_ports(component_library, component_name):
        if candidate_name == port_name:
            return candidate_type
    return None


def _output_drivers(
    design: UHIRDesign,
    component_library: dict[str, dict[str, Any]] | None,
) -> dict[str, str]:
    producer_to_instance = _producer_instance_map(design)
    producer_to_register = _producer_register_map(design)
    returned_values: list[str] = []
    for region in design.regions:
        if region.parent is not None:
            continue
        for node in region.nodes:
            if node.opcode == "ret" and node.operands:
                returned_values.append(node.operands[0])
    drivers: dict[str, str] = {}
    for output, returned in zip(design.outputs, returned_values, strict=False):
        if returned in producer_to_register:
            drivers[output.name] = producer_to_register[returned]
        elif returned in producer_to_instance:
            drivers[output.name] = _instance_result_signal(design, producer_to_instance[returned], component_library)
        else:
            drivers[output.name] = returned
    return drivers


def _lowered_data_inputs(design: UHIRDesign, memory_interfaces: dict[str, dict[str, Any]]) -> list[UHIRPort]:
    lowered: list[UHIRPort] = []
    for port in design.inputs:
        if not _is_memref_type(port.type) or port.name not in memory_interfaces:
            lowered.append(UHIRPort(port.direction, port.name, port.type))
            continue
        interface = memory_interfaces[port.name]
        if interface["read_type"] is not None:
            lowered.append(UHIRPort("input", f"{port.name}_rdata", interface["read_type"]))
    return lowered


def _lowered_data_outputs(design: UHIRDesign, memory_interfaces: dict[str, dict[str, Any]]) -> list[UHIRPort]:
    lowered: list[UHIRPort] = []
    for port in design.outputs:
        if not _is_memref_type(port.type) or port.name not in memory_interfaces:
            lowered.append(UHIRPort(port.direction, port.name, port.type))
    seen_memories: set[str] = set()
    for port in [*design.inputs, *design.outputs]:
        if not _is_memref_type(port.type) or port.name not in memory_interfaces or port.name in seen_memories:
            continue
        seen_memories.add(port.name)
        interface = memory_interfaces[port.name]
        if interface["addr_type"] is not None:
            lowered.append(UHIRPort("output", f"{port.name}_addr", interface["addr_type"]))
        if interface["has_write"] and interface["write_type"] is not None:
            lowered.append(UHIRPort("output", f"{port.name}_wdata", interface["write_type"]))
        if interface["has_write"]:
            lowered.append(UHIRPort("output", f"{port.name}_we", "i1"))
    return lowered


def _memory_interface_assigns(memory_interfaces: dict[str, dict[str, Any]]) -> list[UHIRAssign]:
    assigns: list[UHIRAssign] = []
    for interface in memory_interfaces.values():
        memory_name = interface["memory_name"]
        instance_id = interface["instance_id"]
        if interface["addr_type"] is not None:
            assigns.append(UHIRAssign(f"{memory_name}_addr", f"{instance_id}_addr"))
        if interface["has_write"] and interface["write_type"] is not None:
            assigns.append(UHIRAssign(f"{memory_name}_wdata", f"{instance_id}_wdata"))
        if interface["has_write"]:
            assigns.append(UHIRAssign(f"{memory_name}_we", f"{instance_id}_we"))
        if interface["read_type"] is not None:
            assigns.append(UHIRAssign(f"{instance_id}_rdata", f"{memory_name}_rdata"))
    return assigns


def _memory_interfaces(
    design: UHIRDesign,
    component_library: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if component_library is None:
        return {}

    top_level_memories = {
        port.name: port.type
        for port in [*design.inputs, *design.outputs]
        if _is_memref_type(port.type)
    }
    instance_components = {
        resource.id: resource.value
        for resource in design.resources
        if resource.kind == "fu" and _component_kind(component_library, resource.value) == "memory"
    }
    interfaces: dict[str, dict[str, Any]] = {}

    for region in design.regions:
        for node in region.nodes:
            instance_id = node.attributes.get("bind")
            if not isinstance(instance_id, str) or instance_id not in instance_components:
                continue
            if not node.operands:
                raise ValueError(f"memory-bound node '{node.id}' must declare the memory name as its first operand")
            memory_name = node.operands[0]
            if memory_name not in top_level_memories:
                raise ValueError(
                    f"memory-bound node '{node.id}' references memory '{memory_name}' that is not a top-level memref port"
                )
            component_name = instance_components[instance_id]
            existing = interfaces.get(memory_name)
            if existing is None:
                interfaces[memory_name] = {
                    "memory_name": memory_name,
                    "component_name": component_name,
                    "instance_id": instance_id,
                    "has_read": False,
                    "has_write": False,
                    "addr_type": _memory_port_type(component_library, component_name, "addr"),
                    "write_type": _memory_port_type(component_library, component_name, "wdata"),
                    "read_type": _memory_port_type(component_library, component_name, "rdata"),
                }
                existing = interfaces[memory_name]
            elif existing["instance_id"] != instance_id:
                raise ValueError(
                    f"uglir lowering currently expects at most one memory interface instance per top-level memref; "
                    f"memory '{memory_name}' uses both '{existing['instance_id']}' and '{instance_id}'"
                )
            if node.opcode == "load":
                existing["has_read"] = True
            elif node.opcode == "store":
                existing["has_write"] = True

    for interface in interfaces.values():
        if interface["has_read"] and interface["read_type"] is None:
            raise ValueError(
                f"memory component '{interface['component_name']}' for '{interface['memory_name']}' must declare an output read-data port"
            )
        if (interface["has_read"] or interface["has_write"]) and interface["addr_type"] is None:
            raise ValueError(
                f"memory component '{interface['component_name']}' for '{interface['memory_name']}' must declare an address input port"
            )
        if interface["has_write"] and interface["write_type"] is None:
            raise ValueError(
                f"memory component '{interface['component_name']}' for '{interface['memory_name']}' must declare a write-data input port"
            )
    return interfaces


def _validate_memory_interface_schedule(controller, memory_interfaces: dict[str, dict[str, Any]]) -> None:
    instance_to_memory = {
        interface["instance_id"]: interface["memory_name"]
        for interface in memory_interfaces.values()
    }
    if not instance_to_memory:
        return
    for emit in controller.emits:
        memory_issues: dict[str, list[str]] = {}
        for issue in _iter_issue_actions(emit.attributes):
            instance_id, _, node_id = issue.partition("<-")
            memory_name = instance_to_memory.get(instance_id)
            if memory_name is None:
                continue
            memory_issues.setdefault(memory_name, []).append(node_id or issue)
        for memory_name, node_ids in memory_issues.items():
            distinct_node_ids = sorted(set(node_ids))
            if len(distinct_node_ids) > 1:
                joined = ", ".join(distinct_node_ids)
                raise ValueError(
                    "uglir lowering currently supports only one access per memory interface per FSMD state; "
                    f"state '{emit.state}' issues competing accesses to memory '{memory_name}' via: {joined}"
                )


def _operand_port_expr(
    design: UHIRDesign,
    controller,
    resource_id: str,
    component_name: str,
    port_name: str,
    port_type: str,
    state_code: dict[str, int],
    component_library: dict[str, dict[str, Any]],
) -> str:
    node_by_id = {node.id: node for region in design.regions for node in region.nodes}
    branches: list[tuple[str, str]] = []
    for emit in controller.emits:
        for issue in _iter_issue_actions(emit.attributes):
            instance, _, node_id = issue.partition("<-")
            if instance != resource_id or not node_id:
                continue
            node = node_by_id.get(node_id)
            if node is None:
                continue
            binding_key = _component_port_binding(component_library, component_name, node.opcode, port_name)
            if binding_key is None:
                continue
            source_expr = _binding_key_expr(design, node, binding_key, component_library)
            branches.append((emit.state, source_expr))

    if not branches:
        return _default_net_expr(port_type)
    unique_exprs = {expr for _, expr in branches}
    if len(unique_exprs) == 1:
        return next(iter(unique_exprs))
    ordered = sorted(branches, key=lambda item: state_code[item[0]])
    parts = [f"{_state_eq_expr(state_code[state])} ? {expr}" for state, expr in ordered]
    return " : ".join(parts) + f" : {_default_net_expr(port_type)}"


def _instance_port_names(component_library: dict[str, dict[str, Any]], component_name: str) -> tuple[str, ...]:
    return tuple(port_name for port_name, _ in _instance_ports(component_library, component_name))


def _instance_ports(component_library: dict[str, dict[str, Any]], component_name: str) -> tuple[tuple[str, str], ...]:
    component = component_library.get(component_name)
    if component is None:
        raise ValueError(f"component library does not define component '{component_name}'")
    ports = component.get("ports")
    if not isinstance(ports, dict):
        raise ValueError(f"component '{component_name}' must define object-valued 'ports'")
    normalized: list[tuple[str, str]] = []
    for port_name, port_payload in ports.items():
        if not isinstance(port_payload, dict):
            raise ValueError(f"component '{component_name}' port '{port_name}' must be an object")
        port_type = port_payload.get("type")
        if not isinstance(port_type, str) or not port_type:
            raise ValueError(f"component '{component_name}' port '{port_name}' must define string 'type'")
        normalized.append((str(port_name), port_type))
    return tuple(normalized)


def _instance_input_ports(component_library: dict[str, dict[str, Any]], component_name: str) -> tuple[tuple[str, str], ...]:
    component = component_library.get(component_name)
    if component is None:
        raise ValueError(f"component library does not define component '{component_name}'")
    ports = component.get("ports")
    if not isinstance(ports, dict):
        raise ValueError(f"component '{component_name}' must define object-valued 'ports'")
    inputs: list[tuple[str, str]] = []
    for port_name, port_payload in ports.items():
        if not isinstance(port_payload, dict):
            raise ValueError(f"component '{component_name}' port '{port_name}' must be an object")
        direction = port_payload.get("dir")
        port_type = port_payload.get("type")
        if direction == "input" and isinstance(port_type, str) and port_type:
            inputs.append((str(port_name), port_type))
    return tuple(inputs)


def _instance_output_ports(component_library: dict[str, dict[str, Any]], component_name: str) -> tuple[tuple[str, str], ...]:
    component = component_library.get(component_name)
    if component is None:
        raise ValueError(f"component library does not define component '{component_name}'")
    ports = component.get("ports")
    if not isinstance(ports, dict):
        raise ValueError(f"component '{component_name}' must define object-valued 'ports'")
    outputs: list[tuple[str, str]] = []
    for port_name, port_payload in ports.items():
        if not isinstance(port_payload, dict):
            raise ValueError(f"component '{component_name}' port '{port_name}' must be an object")
        direction = port_payload.get("dir")
        port_type = port_payload.get("type")
        if direction == "output" and isinstance(port_type, str) and port_type:
            outputs.append((str(port_name), port_type))
    return tuple(outputs)


def _instance_result_signal(
    design: UHIRDesign,
    instance_id: str,
    component_library: dict[str, dict[str, Any]] | None,
) -> str:
    if component_library is None:
        return f"{instance_id}_y"
    component_name = _resource_value(design, instance_id)
    output_ports = _instance_output_ports(component_library, component_name)
    for preferred in ("y", "rdata"):
        for port_name, _ in output_ports:
            if port_name == preferred:
                return f"{instance_id}_{port_name}"
    if output_ports:
        return f"{instance_id}_{output_ports[0][0]}"
    raise ValueError(f"component '{component_name}' bound at '{instance_id}' has no output port")


def _component_port_binding(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
    opcode_name: str,
    port_name: str,
) -> str | None:
    component = component_library.get(component_name)
    if component is None:
        raise ValueError(f"component library does not define component '{component_name}'")
    supports = component.get("supports")
    if not isinstance(supports, dict):
        raise ValueError(f"component '{component_name}' must define object-valued 'supports'")
    support = supports.get(opcode_name)
    if support is None:
        raise ValueError(f"component '{component_name}' does not support opcode '{opcode_name}'")
    if not isinstance(support, dict):
        raise ValueError(f"component '{component_name}' support '{opcode_name}' must be an object")
    binding = support.get("bind")
    if binding is None:
        return None
    if not isinstance(binding, dict):
        raise ValueError(f"component '{component_name}' support '{opcode_name}' must use object-valued 'bind'")
    binding_key = binding.get(port_name)
    if binding_key is None:
        return None
    if not isinstance(binding_key, str):
        raise ValueError(f"component '{component_name}' support '{opcode_name}' bind '{port_name}' must be a string")
    return binding_key


def _binding_key_expr(
    design: UHIRDesign,
    node,
    binding_key: str,
    component_library: dict[str, dict[str, Any]] | None,
) -> str:
    if binding_key.startswith("operand"):
        operand_index_text = binding_key[len("operand") :]
        if not operand_index_text.isdigit():
            raise ValueError(f"invalid operand binding key '{binding_key}'")
        operand_index = int(operand_index_text)
        if operand_index >= len(node.operands):
            raise ValueError(
                f"node '{node.id}' opcode '{node.opcode}' does not provide operand index {operand_index} for binding '{binding_key}'"
            )
        return _resolve_value_signal(design, node.operands[operand_index], component_library)
    return binding_key


def _resolve_value_signal(
    design: UHIRDesign,
    value_id: str,
    component_library: dict[str, dict[str, Any]] | None,
) -> str:
    producer_to_register = _producer_register_map(design)
    if value_id in producer_to_register:
        return producer_to_register[value_id]
    producer_to_instance = _producer_instance_map(design)
    instance = producer_to_instance.get(value_id)
    if instance is not None:
        return _instance_result_signal(design, instance, component_library)
    return value_id


def _default_net_expr(port_type: str) -> str:
    if port_type == "i1":
        return "false"
    return f"0:{port_type}"


def _opcode_expr(
    design: UHIRDesign,
    controller,
    resource_id: str,
    component_name: str,
    state_code: dict[str, int],
    component_library: dict[str, dict[str, Any]],
) -> str:
    component = component_library.get(component_name)
    if component is None:
        raise ValueError(f"component library does not define component '{component_name}'")
    supports = component.get("supports")
    if not isinstance(supports, dict):
        raise ValueError(f"component '{component_name}' must define object-valued 'supports'")

    node_opcode = {node.id: node.opcode for region in design.regions for node in region.nodes}
    state_to_opcode: dict[str, int] = {}
    for emit in controller.emits:
        for issue in _iter_issue_actions(emit.attributes):
            instance, _, node_id = issue.partition("<-")
            if instance != resource_id or not node_id:
                continue
            opcode_name = node_opcode.get(node_id)
            if opcode_name is None:
                continue
            support = supports.get(opcode_name)
            if not isinstance(support, dict):
                raise ValueError(f"component '{component_name}' does not support opcode '{opcode_name}'")
            opcode_literal = support.get("opcode")
            if not isinstance(opcode_literal, int):
                raise ValueError(f"component '{component_name}' support '{opcode_name}' must define integer 'opcode'")
            state_to_opcode[emit.state] = opcode_literal

    if not state_to_opcode:
        return "0"
    unique_opcodes = set(state_to_opcode.values())
    if len(unique_opcodes) == 1:
        return str(next(iter(unique_opcodes)))
    branches = [
        f"{_state_eq_expr(state_code[state])} ? {opcode}"
        for state, opcode in sorted(state_to_opcode.items(), key=lambda item: state_code[item[0]])
    ]
    return " : ".join(branches) + " : 0"


def _iter_issue_actions(attributes: dict[str, Any]) -> tuple[str, ...]:
    raw_issues = attributes.get("issue", ())
    normalized: list[str] = []
    for raw_issue in raw_issues:
        if not isinstance(raw_issue, str):
            continue
        for item in raw_issue.split(","):
            issue = item.strip()
            if issue:
                normalized.append(issue)
    return tuple(normalized)
