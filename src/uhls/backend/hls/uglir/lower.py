"""FSM-to-uglir lowering entrypoint."""

from __future__ import annotations

from math import ceil, log2
import re
from typing import Any

from uhls.backend.hls.uhir.model import (
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
    if not design.controllers:
        raise ValueError("uglir lowering expects at least one fsm controller")
    top_controller = _require_top_level_controller(design)
    controller_codes = {
        controller.name: {state.name: state.attributes["code"] for state in controller.states}
        for controller in design.controllers
    }
    lowered = UHIRDesign(name=design.name, stage="uglir")
    helper_signal_ids: dict[tuple[str, str], str] = {}
    memory_interfaces = _memory_interfaces(design, component_library)
    _validate_memory_interface_schedule(top_controller, memory_interfaces)
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

    lowered.resources.append(UHIRResource("net", "req_fire", "i1"))
    lowered.resources.append(UHIRResource("net", "resp_fire", "i1"))
    for interface in memory_interfaces.values():
        lowered.resources.append(UHIRResource("port", interface["memory_name"], interface["component_name"], interface["memory_name"]))
    for controller in design.controllers:
        state_type = _state_type(controller)
        lowered.resources.append(UHIRResource("reg", _controller_state_id(controller, top_controller), state_type))
        lowered.resources.append(UHIRResource("net", _controller_next_state_id(controller, top_controller), state_type))
        for port in controller.inputs:
            signal_id = _controller_port_signal_id(controller, port.name, top_controller)
            if signal_id not in {"req_valid", "resp_ready"}:
                lowered.resources.append(UHIRResource("net", signal_id, port.type))
        for port in controller.outputs:
            signal_id = _controller_port_signal_id(controller, port.name, top_controller)
            if signal_id not in {"req_ready", "resp_valid"}:
                lowered.resources.append(UHIRResource("net", signal_id, port.type))
    for signal_name in _link_export_signal_names(design, top_controller):
        lowered.resources.append(UHIRResource("net", signal_name, "i1"))

    for resource in design.resources:
        if resource.kind == "fu":
            component_kind = None if component_library is None else _component_kind(component_library, resource.value)
            if component_kind != "memory":
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

    latch_targets = sorted(
        {
            register
            for controller in design.controllers
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
        ]
    )
    for controller in design.controllers:
        for port in controller.outputs:
            lowered.assigns.append(
                UHIRAssign(
                    _controller_port_signal_id(controller, port.name, top_controller),
                    _controller_output_expr(controller, port.name, controller_codes[controller.name], top_controller),
                )
            )
        lowered.assigns.append(
            UHIRAssign(
                _controller_next_state_id(controller, top_controller),
                _next_state_expr(controller, controller_codes[controller.name], top_controller),
            )
        )
    for assign in _controller_link_assigns(design, top_controller, controller_codes):
        lowered.assigns.append(assign)

    for resource in design.resources:
        if resource.kind != "fu":
            continue
        if component_library is None:
            lowered.assigns.append(UHIRAssign(f"{resource.id}_go", _issue_expr(design.controllers, top_controller, resource.id, controller_codes)))
            lowered.attachments.append(UHIRAttach(resource.id, "go", f"{resource.id}_go"))
            if any(candidate.id == f"{resource.id}_y" for candidate in lowered.resources):
                lowered.attachments.append(UHIRAttach(resource.id, "y", f"{resource.id}_y"))
            continue

        if _component_kind(component_library, resource.value) == "memory":
            for port_name, port_type in _instance_input_ports(component_library, resource.value):
                if port_name == "op" or port_name in _component_issue_bindings(component_library, resource.value):
                    continue
                _lower_explicit_input_mux(
                    lowered,
                    design,
                    top_controller,
                    controller_codes,
                    component_library,
                    helper_signal_ids,
                    resource.id,
                    resource.value,
                    port_name,
                    port_type,
                )
            continue

        port_names = _instance_port_names(component_library, resource.value)
        issue_bindings = _component_issue_bindings(component_library, resource.value)
        for port_name, binding_key in issue_bindings.items():
            if port_name not in port_names:
                raise ValueError(
                    f"component '{resource.value}' issue binding references unknown port '{port_name}'"
                )
            lowered.assigns.append(
                UHIRAssign(
                    f"{resource.id}_{port_name}",
                    _issue_port_expr(
                        design.controllers,
                        top_controller,
                        resource.id,
                        _port_type(component_library, resource.value, port_name),
                        binding_key,
                        controller_codes,
                    ),
                )
            )
        if "op" in port_names:
            lowered.assigns.append(
                UHIRAssign(
                    f"{resource.id}_op",
                    _opcode_expr(design, design.controllers, top_controller, resource.id, resource.value, controller_codes, component_library),
                )
            )
        for port_name, port_type in _instance_input_ports(component_library, resource.value):
            if port_name == "op" or port_name in issue_bindings:
                continue
            _lower_explicit_input_mux(
                lowered,
                design,
                top_controller,
                controller_codes,
                component_library,
                helper_signal_ids,
                resource.id,
                resource.value,
                port_name,
                port_type,
            )
        for port_name in port_names:
            lowered.attachments.append(UHIRAttach(resource.id, port_name, f"{resource.id}_{port_name}"))

    for register in latch_targets:
        lowered.assigns.append(UHIRAssign(f"latch_{register}", _latch_expr(design.controllers, top_controller, register, controller_codes)))
        lowered.assigns.append(
            UHIRAssign(
                f"sel_{register}",
                _select_expr(design, design.controllers, top_controller, register, controller_codes, component_library),
            )
        )
        lowered.glue_muxes.append(_build_register_mux(design, register, component_library))

    for assign in _memory_interface_assigns(memory_interfaces):
        lowered.assigns.append(assign)

    for output_name, driver in _output_drivers(design, component_library).items():
        lowered.assigns.append(UHIRAssign(output_name, driver))

    top_seq_block = UHIRSeqBlock(
        clock="clk",
        reset="rst",
        reset_updates=[UHIRSeqUpdate(_controller_state_id(top_controller, top_controller), str(controller_codes[top_controller.name]["IDLE"]))],
        updates=[UHIRSeqUpdate(_controller_state_id(top_controller, top_controller), _controller_next_state_id(top_controller, top_controller))],
    )
    for register in latch_targets:
        top_seq_block.updates.append(UHIRSeqUpdate(register, f"mx_{register}", f"latch_{register}"))
    lowered.seq_blocks.append(top_seq_block)
    for controller in design.controllers:
        if controller.name == top_controller.name:
            continue
        lowered.seq_blocks.append(
            UHIRSeqBlock(
                clock="clk",
                reset="rst",
                reset_updates=[UHIRSeqUpdate(_controller_state_id(controller, top_controller), str(controller_codes[controller.name]["IDLE"]))],
                updates=[UHIRSeqUpdate(_controller_state_id(controller, top_controller), _controller_next_state_id(controller, top_controller))],
            )
        )
    return _apply_signal_naming_convention(lowered)


def _apply_signal_naming_convention(design: UHIRDesign) -> UHIRDesign:
    rename_map = {
        resource.id: _uglir_signal_name(resource.kind, resource.id)
        for resource in design.resources
        if resource.kind in {"reg", "net", "mux"}
    }
    if not rename_map:
        return design

    normalized = UHIRDesign(name=design.name, stage=design.stage)
    normalized.inputs = list(design.inputs)
    normalized.outputs = list(design.outputs)
    normalized.constants = list(design.constants)
    normalized.schedule = design.schedule

    for resource in design.resources:
        normalized.resources.append(
            UHIRResource(
                resource.kind,
                rename_map.get(resource.id, resource.id),
                resource.value,
                resource.target,
            )
        )
    for assign in design.assigns:
        normalized.assigns.append(
            UHIRAssign(
                rename_map.get(assign.target, assign.target),
                _rewrite_signal_expr(assign.expr, rename_map),
            )
        )
    for attachment in design.attachments:
        normalized.attachments.append(
            UHIRAttach(
                attachment.instance,
                attachment.port,
                rename_map.get(attachment.signal, attachment.signal),
            )
        )
    for glue_mux in design.glue_muxes:
        normalized.glue_muxes.append(
            UHIRGlueMux(
                name=rename_map.get(glue_mux.name, glue_mux.name),
                type=glue_mux.type,
                select=rename_map.get(glue_mux.select, glue_mux.select),
                cases=[
                    UHIRGlueMuxCase(case.key, rename_map.get(case.source, case.source))
                    for case in glue_mux.cases
                ],
            )
        )
    for seq_block in design.seq_blocks:
        normalized.seq_blocks.append(
            UHIRSeqBlock(
                clock=seq_block.clock,
                reset=None if seq_block.reset is None else _rewrite_signal_expr(seq_block.reset, rename_map),
                reset_updates=[
                    UHIRSeqUpdate(
                        rename_map.get(update.target, update.target),
                        _rewrite_signal_expr(update.value, rename_map),
                        None if update.enable is None else _rewrite_signal_expr(update.enable, rename_map),
                    )
                    for update in seq_block.reset_updates
                ],
                updates=[
                    UHIRSeqUpdate(
                        rename_map.get(update.target, update.target),
                        _rewrite_signal_expr(update.value, rename_map),
                        None if update.enable is None else _rewrite_signal_expr(update.enable, rename_map),
                    )
                    for update in seq_block.updates
                ],
            )
        )
    return normalized


def _uglir_signal_name(kind: str, signal_id: str) -> str:
    if kind == "reg":
        return f"{signal_id}_q"
    if kind in {"net", "mux"}:
        return f"{signal_id}_n"
    return signal_id


def _rewrite_signal_expr(expr: str, rename_map: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return rename_map.get(token, token)

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", replace, expr)


def _state_type(controller) -> str:
    encoding = controller.attributes.get("encoding")
    count = max(len(controller.states), 1)
    if encoding == "one_hot":
        return f"u{count}"
    width = max(1, ceil(log2(count)))
    return f"u{width}"


def _require_top_level_controller(design: UHIRDesign):
    top_level = [controller for controller in design.controllers if controller.attributes.get("protocol") == "req_resp"]
    if len(top_level) != 1:
        raise ValueError("uglir lowering currently expects exactly one top-level req_resp controller")
    return top_level[0]


def _state_eq_expr(state_signal: str, code: int) -> str:
    return f"{state_signal} == {code}"


def _controller_state_id(controller, top_controller) -> str:
    return "state" if controller.name == top_controller.name else f"{controller.name}_state"


def _controller_next_state_id(controller, top_controller) -> str:
    return "next_state" if controller.name == top_controller.name else f"{controller.name}_next_state"


def _controller_port_signal_id(controller, port_name: str, top_controller) -> str:
    if controller.name == top_controller.name and port_name in {"req_valid", "resp_ready", "req_ready", "resp_valid"}:
        return port_name
    return f"{controller.name}_{port_name}"


def _rewrite_controller_expr(expr: str, controller, top_controller) -> str:
    port_map = {
        port.name: _controller_port_signal_id(controller, port.name, top_controller)
        for port in [*controller.inputs, *controller.outputs]
    }

    def replace(match: re.Match[str]) -> str:
        name = match.group(0)
        return port_map.get(name, name)

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", replace, expr)


def _next_state_expr(controller, state_code: dict[str, int], top_controller) -> str:
    state_signal = _controller_state_id(controller, top_controller)
    branches: list[str] = []
    for transition in controller.transitions:
        condition = transition.attributes.get("when")
        state_guard = _state_eq_expr(state_signal, state_code[transition.source])
        transition_condition = _normalized_transition_condition(condition, controller, top_controller)
        if transition_condition is None:
            branches.append(f"{state_guard} ? {state_code[transition.target]}")
        else:
            branches.append(f"({state_guard} && {transition_condition}) ? {state_code[transition.target]}")
    return " : ".join(branches) + f" : {state_code['IDLE']}"


def _normalized_transition_condition(condition: object, controller, top_controller) -> str | None:
    if not isinstance(condition, str) or not condition:
        return None
    rewritten = _rewrite_controller_expr(condition, controller, top_controller).strip()
    if rewritten in {"true", "(true)"}:
        return None
    normalized = _strip_one_outer_paren_pair(rewritten)
    if controller.name == top_controller.name:
        if normalized == "req_valid && req_ready":
            return "req_fire"
        if normalized == "resp_valid && resp_ready":
            return "resp_fire"
    return normalized


def _strip_one_outer_paren_pair(expr: str) -> str:
    if not (expr.startswith("(") and expr.endswith(")")):
        return expr
    depth = 0
    for index, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(expr) - 1:
                return expr
    return expr[1:-1].strip()


def _issue_expr(controllers, top_controller, resource_id: str, controller_codes: dict[str, dict[str, int]]) -> str:
    active_states = [
        _state_eq_expr(_controller_state_id(controller, top_controller), controller_codes[controller.name][emit.state])
        for controller in controllers
        for emit in controller.emits
        for attrs in (emit.attributes,)
        for issue in _iter_issue_actions(attrs)
        if issue.split("<-", 1)[0] == resource_id
    ]
    if not active_states:
        return "false"
    return " | ".join(active_states)


def _issue_port_expr(
    controllers,
    top_controller,
    resource_id: str,
    port_type: str,
    binding_key: str,
    controller_codes: dict[str, dict[str, int]],
) -> str:
    active_condition = _issue_expr(controllers, top_controller, resource_id, controller_codes)
    if active_condition == "false":
        return _default_net_expr(port_type)
    if binding_key == "true" and _default_net_expr(port_type) == "false":
        return active_condition
    return f"({active_condition}) ? {binding_key} : {_default_net_expr(port_type)}"


def _latch_expr(controllers, top_controller, register: str, controller_codes: dict[str, dict[str, int]]) -> str:
    active_states = [
        _state_eq_expr(_controller_state_id(controller, top_controller), controller_codes[controller.name][emit.state])
        for controller in controllers
        for emit in controller.emits
        if register in emit.attributes.get("latch", ())
    ]
    if not active_states:
        return "false"
    return " | ".join(active_states)


def _select_expr(
    design: UHIRDesign,
    controllers,
    top_controller,
    register: str,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]] | None = None,
) -> str:
    choices = _register_state_sources(design, controllers, top_controller, register, controller_codes, component_library)
    labels = _register_mux_case_labels(register, [source for _, source in choices])
    if not choices:
        return "HOLD"
    branches = [f"{condition} ? {labels[source]}" for condition, source in choices]
    return " : ".join(branches) + " : HOLD"


def _build_register_mux(
    design: UHIRDesign,
    register: str,
    component_library: dict[str, dict[str, Any]] | None,
) -> UHIRGlueMux:
    register_type = _resource_value(design, register)
    glue_mux = UHIRGlueMux(name=f"mx_{register}", type=register_type, select=f"sel_{register}")
    sources = [register, *_register_possible_sources(design, register, component_library)]
    labels = _register_mux_case_labels(register, sources)
    glue_mux.cases.append(UHIRGlueMuxCase("HOLD", register))
    seen_sources = {register}
    for source in _register_possible_sources(design, register, component_library):
        if source in seen_sources:
            continue
        glue_mux.cases.append(UHIRGlueMuxCase(labels[source], source))
        seen_sources.add(source)
    return glue_mux


def _controller_region_ids(controller, design: UHIRDesign) -> tuple[str, ...]:
    roots: list[str] = []
    region = controller.attributes.get("region")
    if isinstance(region, str) and region:
        roots.append(region)
    false_region = controller.attributes.get("false_region")
    if isinstance(false_region, str) and false_region:
        roots.append(false_region)
    children_by_parent: dict[str, list[str]] = {}
    for region in design.regions:
        if region.parent is None:
            continue
        children_by_parent.setdefault(region.parent, []).append(region.id)
    ordered: list[str] = []
    seen: set[str] = set()

    def visit(region_id: str) -> None:
        if region_id in seen:
            return
        seen.add(region_id)
        ordered.append(region_id)
        for child_id in children_by_parent.get(region_id, ()):
            visit(child_id)

    for root in roots:
        visit(root)
    return tuple(ordered)


def _controller_output_expr(controller, port_name: str, state_code: dict[str, int], top_controller) -> str:
    active_states = [
        _state_eq_expr(_controller_state_id(controller, top_controller), state_code[emit.state])
        for emit in controller.emits
        if emit.attributes.get(port_name) is True
    ]
    if not active_states:
        return "false"
    return " | ".join(active_states)


def _controller_action_expr(
    controller,
    top_controller,
    action_name: str,
    node_id: str,
    state_code: dict[str, int],
) -> str:
    active_states = [
        _state_eq_expr(_controller_state_id(controller, top_controller), state_code[emit.state])
        for emit in controller.emits
        if node_id in emit.attributes.get(action_name, ())
    ]
    if not active_states:
        return "false"
    return " | ".join(active_states)


def _controller_signal_ref(controller, top_controller, signal_name: str) -> str:
    known_ports = {port.name for port in [*controller.inputs, *controller.outputs]}
    if signal_name in known_ports:
        return _controller_port_signal_id(controller, signal_name, top_controller)
    return signal_name


def _link_export_signal_names(design: UHIRDesign, top_controller) -> list[str]:
    declared = {
        port.name
        for port in [*design.inputs, *design.outputs]
    }
    declared.update(constant.name for constant in design.constants)
    declared.update(resource.id for resource in design.resources)
    exported: list[str] = []
    seen: set[str] = set()
    for controller in design.controllers:
        known_ports = {port.name for port in [*controller.inputs, *controller.outputs]}
        for link in controller.links:
            for attr_name in ("ready", "done"):
                mapping = link.attributes.get(attr_name)
                if not isinstance(mapping, tuple) or len(mapping) != 2:
                    continue
                parent_signal = mapping[0]
                if not isinstance(parent_signal, str) or not parent_signal:
                    continue
                if parent_signal in declared or parent_signal in known_ports or parent_signal in seen:
                    continue
                seen.add(parent_signal)
                exported.append(parent_signal)
    return exported


def _controller_link_assigns(design: UHIRDesign, top_controller, controller_codes: dict[str, dict[str, int]]) -> list[UHIRAssign]:
    controllers_by_name = {controller.name: controller for controller in design.controllers}
    assigns: list[UHIRAssign] = []
    for parent in design.controllers:
        for link in parent.links:
            child = controllers_by_name.get(link.child)
            if child is None:
                raise ValueError(f"uglir lowering references unknown child controller '{link.child}'")
            act_mapping = link.attributes.get("act")
            if isinstance(act_mapping, tuple) and len(act_mapping) == 2:
                _, child_input = act_mapping
                assigns.append(
                    UHIRAssign(
                        _controller_port_signal_id(child, child_input, top_controller),
                        _controller_action_expr(parent, top_controller, "activate", link.node, controller_codes[parent.name]),
                    )
                )
            ready_mapping = link.attributes.get("ready")
            if isinstance(ready_mapping, tuple) and len(ready_mapping) == 2:
                parent_signal, child_output = ready_mapping
                assigns.append(
                    UHIRAssign(
                        str(parent_signal),
                        _controller_port_signal_id(child, str(child_output), top_controller),
                    )
                )
            done_mapping = link.attributes.get("done")
            if isinstance(done_mapping, tuple) and len(done_mapping) == 2:
                parent_signal, child_output = done_mapping
                assigns.append(
                    UHIRAssign(
                        str(parent_signal),
                        _controller_port_signal_id(child, str(child_output), top_controller),
                    )
                )
            done_ready_mapping = link.attributes.get("done_ready")
            if isinstance(done_ready_mapping, tuple) and len(done_ready_mapping) == 2:
                parent_signal, child_input = done_ready_mapping
                assigns.append(
                    UHIRAssign(
                        _controller_port_signal_id(child, str(child_input), top_controller),
                        _controller_signal_ref(parent, top_controller, str(parent_signal)),
                    )
                )
    return assigns


def _register_possible_sources(
    design: UHIRDesign,
    register: str,
    component_library: dict[str, dict[str, Any]] | None,
) -> list[str]:
    sources: list[str] = []
    for region in design.regions:
        for binding in region.value_bindings:
            if binding.register != register:
                continue
            source = _resolve_producer_signal(design, binding.producer, component_library)
            if source is not None:
                sources.append(source)
    return sources


def _register_state_sources(
    design: UHIRDesign,
    controllers,
    top_controller,
    register: str,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]] | None,
) -> list[tuple[str, str]]:
    region_by_id = {region.id: region for region in design.regions}
    choices: list[tuple[str, str]] = []
    for controller in controllers:
        if not controller.attributes.get("region"):
            continue
        live_start_to_source: dict[int, str] = {}
        for region_id in _controller_region_ids(controller, design):
            region = region_by_id.get(region_id)
            if region is None:
                continue
            for binding in region.value_bindings:
                if binding.register != register:
                    continue
                source = _resolve_producer_signal(design, binding.producer, component_library)
                if source is None:
                    continue
                live_start_to_source[binding.live_start] = source
        for emit in controller.emits:
            if register not in emit.attributes.get("latch", ()):
                continue
            if emit.state.startswith("T"):
                time_step = int(emit.state[1:])
                source = live_start_to_source.get(time_step)
                if source is not None:
                    condition = _state_eq_expr(
                        _controller_state_id(controller, top_controller),
                        controller_codes[controller.name][emit.state],
                    )
                    choices.append((condition, source))
    return choices


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


def _port_type(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
    port_name: str,
) -> str:
    port_type = _memory_port_type(component_library, component_name, port_name)
    if port_type is None:
        raise ValueError(f"component '{component_name}' does not define port '{port_name}'")
    return port_type


def _output_drivers(
    design: UHIRDesign,
    component_library: dict[str, dict[str, Any]] | None,
) -> dict[str, str]:
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
        else:
            drivers[output.name] = _resolve_value_signal(design, returned, component_library)
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
            memory_issues.setdefault(memory_name, []).append(_issued_node_base_id(node_id or issue))
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
    controllers,
    top_controller,
    resource_id: str,
    component_name: str,
    port_name: str,
    port_type: str,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]],
) -> str:
    node_by_id = {node.id: node for region in design.regions for node in region.nodes}
    branches: list[tuple[str, str]] = []
    for controller in controllers:
        for emit in controller.emits:
            for issue in _iter_issue_actions(emit.attributes):
                instance, _, node_id = issue.partition("<-")
                if instance != resource_id or not node_id:
                    continue
                node = node_by_id.get(_issued_node_base_id(node_id))
                if node is None:
                    continue
                binding_key = _component_port_binding(component_library, component_name, node.opcode, port_name)
                if binding_key is None:
                    continue
                source_expr = _binding_key_expr(design, node, binding_key, component_library)
                condition = _state_eq_expr(_controller_state_id(controller, top_controller), controller_codes[controller.name][emit.state])
                branches.append((condition, source_expr))

    if not branches:
        return _default_net_expr(port_type)
    unique_exprs = {expr for _, expr in branches}
    if len(unique_exprs) == 1:
        return next(iter(unique_exprs))
    parts = [f"{condition} ? {expr}" for condition, expr in branches]
    return " : ".join(parts) + f" : {_default_net_expr(port_type)}"


def _operand_port_choices(
    design: UHIRDesign,
    controllers,
    top_controller,
    resource_id: str,
    component_name: str,
    port_name: str,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    node_by_id = {node.id: node for region in design.regions for node in region.nodes}
    choices: list[tuple[str, str]] = []
    for controller in controllers:
        for emit in controller.emits:
            for issue in _iter_issue_actions(emit.attributes):
                instance, _, node_id = issue.partition("<-")
                if instance != resource_id or not node_id:
                    continue
                node = node_by_id.get(_issued_node_base_id(node_id))
                if node is None:
                    continue
                binding_key = _component_port_binding(component_library, component_name, node.opcode, port_name)
                if binding_key is None:
                    continue
                source_expr = _binding_key_expr(
                    design,
                    node,
                    binding_key,
                    component_library,
                    _issued_node_occurrence_index(node_id),
                )
                condition = _state_eq_expr(
                    _controller_state_id(controller, top_controller),
                    controller_codes[controller.name][emit.state],
                )
                choices.append((condition, source_expr))
    return choices


def _lower_explicit_input_mux(
    lowered: UHIRDesign,
    design: UHIRDesign,
    top_controller,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]],
    helper_signal_ids: dict[tuple[str, str], str],
    resource_id: str,
    component_name: str,
    port_name: str,
    port_type: str,
) -> None:
    target_signal = f"{resource_id}_{port_name}"
    select_signal = f"sel_{resource_id}_{port_name}"
    mux_name = f"mx_{resource_id}_{port_name}"

    lowered.resources.append(UHIRResource("net", select_signal, "ctrl"))
    lowered.resources.append(UHIRResource("mux", mux_name, port_type))

    default_signal = _materialize_glue_source_signal(
        lowered,
        _default_net_expr(port_type),
        port_type,
        helper_signal_ids,
    )
    operand_choices = _operand_port_choices(
        design,
        design.controllers,
        top_controller,
        resource_id,
        component_name,
        port_name,
        controller_codes,
        component_library,
    )
    choices = [
        (
            condition,
            source_expr,
            _materialize_glue_source_signal(lowered, source_expr, port_type, helper_signal_ids),
        )
        for condition, source_expr in operand_choices
    ]
    labels = _input_mux_case_labels(_default_net_expr(port_type), default_signal, choices)

    if choices:
        select_expr = " : ".join(f"{condition} ? {labels[source_signal]}" for condition, _source_expr, source_signal in choices)
        select_expr = f"{select_expr} : {labels[default_signal]}"
    else:
        select_expr = labels[default_signal]
    lowered.assigns.append(UHIRAssign(select_signal, select_expr))
    lowered.assigns.append(UHIRAssign(target_signal, mux_name))

    glue_mux = UHIRGlueMux(name=mux_name, type=port_type, select=select_signal)
    seen_sources: set[str] = set()
    for source_signal in [default_signal, *(source for _, _expr, source in choices)]:
        if source_signal in seen_sources:
            continue
        seen_sources.add(source_signal)
        glue_mux.cases.append(UHIRGlueMuxCase(labels[source_signal], source_signal))
    lowered.glue_muxes.append(glue_mux)


def _materialize_glue_source_signal(
    lowered: UHIRDesign,
    source_expr: str,
    port_type: str,
    helper_signal_ids: dict[tuple[str, str], str],
) -> str:
    if _is_known_uglir_signal(lowered, source_expr):
        return source_expr
    key = (source_expr, port_type)
    existing = helper_signal_ids.get(key)
    if existing is not None:
        return existing
    signal_id = _fresh_helper_signal_id(lowered, source_expr, port_type)
    lowered.resources.append(UHIRResource("net", signal_id, port_type))
    lowered.assigns.append(UHIRAssign(signal_id, source_expr))
    helper_signal_ids[key] = signal_id
    return signal_id


def _is_known_uglir_signal(lowered: UHIRDesign, signal_name: str) -> bool:
    if signal_name in {port.name for port in [*lowered.inputs, *lowered.outputs]}:
        return True
    return any(resource.id == signal_name for resource in lowered.resources if resource.kind in {"reg", "net", "mux"})


def _fresh_helper_signal_id(lowered: UHIRDesign, source_expr: str, port_type: str) -> str:
    prefix = "const" if _looks_like_constant_expr(source_expr) else "src"
    base = f"{prefix}_{port_type.replace('<', '_').replace('>', '_').replace('[', '_').replace(']', '_').replace(':', '_')}"
    existing = {resource.id for resource in lowered.resources}
    index = 0
    while True:
        candidate = f"{base}_{index}"
        if candidate not in existing:
            return candidate
        index += 1


def _looks_like_constant_expr(expr: str) -> bool:
    if expr in {"true", "false"}:
        return True
    return _looks_like_literal(expr)


def _register_mux_case_labels(register: str, sources: list[str]) -> dict[str, str]:
    labels = {register: "HOLD"}
    other_sources = [source for source in sources if source != register]
    labels.update(_stable_source_labels(other_sources))
    return labels


def _input_mux_case_labels(
    default_expr: str,
    default_signal: str,
    choices: list[tuple[str, str, str]],
) -> dict[str, str]:
    ordered_sources = [default_signal, *(source for _condition, _expr, source in choices)]
    seeds: dict[str, str] = {default_signal: _source_label_seed(default_expr)}
    for _condition, source_expr, source_signal in choices:
        seeds.setdefault(source_signal, _source_label_seed(source_expr))
    return _stable_source_labels(ordered_sources, preferred=seeds)


def _stable_source_labels(sources: list[str], preferred: dict[str, str] | None = None) -> dict[str, str]:
    labels: dict[str, str] = {}
    used: dict[str, int] = {}
    for source in sources:
        if source in labels:
            continue
        base = _sanitize_mux_label((preferred or {}).get(source, _source_label_seed(source)))
        count = used.get(base, 0)
        label = base if count == 0 else f"{base}_{count}"
        used[base] = count + 1
        labels[source] = label
    return labels


def _source_label_seed(source: str) -> str:
    if source in {"false", "0:i1"}:
        return "FALSE"
    if source == "true":
        return "TRUE"
    if source.startswith("0:"):
        return "ZERO"
    if ":" in source and _looks_like_literal(source):
        return f"CONST_{source}"
    return f"SRC_{source}"


def _looks_like_literal(text: str) -> bool:
    head, _, _tail = text.partition(":")
    if not head:
        return False
    if head[0] == "-":
        head = head[1:]
    return bool(head) and head.isdigit()


def _sanitize_mux_label(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    sanitized = re.sub(r"_+", "_", sanitized)
    if not sanitized:
        sanitized = "SRC"
    if sanitized[0].isdigit():
        sanitized = f"SRC_{sanitized}"
    return sanitized.upper()


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


def _node_result_signal(
    design: UHIRDesign,
    node,
    component_library: dict[str, dict[str, Any]] | None,
) -> str | None:
    bind = node.attributes.get("bind")
    if not isinstance(bind, str):
        return None
    if component_library is None:
        return f"{bind}_y"
    component_name = _resource_value(design, bind)
    result_port = _component_result_port(component_library, component_name, node.opcode)
    if result_port is not None:
        return f"{bind}_{result_port}"
    return _instance_result_signal(design, bind, component_library)


def _component_result_port(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
    opcode_name: str,
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
    result_ports = [port_name for port_name, binding_key in binding.items() if binding_key == "result"]
    if len(result_ports) > 1:
        raise ValueError(
            f"component '{component_name}' support '{opcode_name}' must map at most one output port to 'result'"
        )
    return result_ports[0] if result_ports else None


def _component_issue_bindings(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
) -> dict[str, str]:
    component = component_library.get(component_name)
    if component is None:
        raise ValueError(f"component library does not define component '{component_name}'")
    issue = component.get("issue")
    if issue is None:
        return {}
    if not isinstance(issue, dict):
        raise ValueError(f"component '{component_name}' must use object-valued 'issue' bindings")
    normalized: dict[str, str] = {}
    for port_name, binding_key in issue.items():
        if not isinstance(binding_key, str) or not binding_key:
            raise ValueError(f"component '{component_name}' issue binding '{port_name}' must be a non-empty string")
        normalized[str(port_name)] = binding_key
    return normalized


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
    occurrence_index: int | None = None,
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
        return _resolve_value_signal(design, node.operands[operand_index], component_library, occurrence_index)
    return binding_key


def _resolve_value_signal(
    design: UHIRDesign,
    value_id: str,
    component_library: dict[str, dict[str, Any]] | None,
    occurrence_index: int | None = None,
) -> str:
    producer_to_register = _producer_register_map(design)
    if value_id in producer_to_register:
        return producer_to_register[value_id]
    result_signal = _resolve_producer_signal(design, value_id, component_library, occurrence_index)
    if result_signal is not None:
        return result_signal
    return value_id


def _resolve_producer_signal(
    design: UHIRDesign,
    value_id: str,
    component_library: dict[str, dict[str, Any]] | None,
    occurrence_index: int | None = None,
) -> str | None:
    producer_node = _producer_node_map(design).get(value_id)
    if producer_node is None:
        return None
    result_signal = _node_result_signal(design, producer_node, component_library)
    if result_signal is not None:
        return result_signal
    return _node_value_expr(design, producer_node, component_library, occurrence_index)


def _node_value_expr(
    design: UHIRDesign,
    node,
    component_library: dict[str, dict[str, Any]] | None,
    occurrence_index: int | None,
) -> str | None:
    if node.opcode == "phi":
        if not node.operands:
            return None
        if occurrence_index is None or occurrence_index > 0:
            chosen = node.operands[1] if len(node.operands) > 1 else node.operands[0]
        else:
            chosen = node.operands[0]
        return _resolve_value_signal(design, chosen, component_library, None if occurrence_index is None else max(occurrence_index - 1, 0))
    if node.opcode == "sel" and len(node.operands) == 3:
        condition = _resolve_value_signal(design, node.operands[0], component_library, occurrence_index)
        true_value = _resolve_value_signal(design, node.operands[1], component_library, occurrence_index)
        false_value = _resolve_value_signal(design, node.operands[2], component_library, occurrence_index)
        return f"{condition} ? {true_value} : {false_value}"
    return None


def _producer_node_map(design: UHIRDesign) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for region in design.regions:
        local_nodes = {node.id: node for node in region.nodes}
        mapping.update(local_nodes)
        for source_map in region.mappings:
            node = local_nodes.get(source_map.node_id)
            if node is not None:
                mapping[source_map.source_id] = node
    return mapping


def _default_net_expr(port_type: str) -> str:
    if port_type == "i1":
        return "false"
    return f"0:{port_type}"


def _opcode_expr(
    design: UHIRDesign,
    controllers,
    top_controller,
    resource_id: str,
    component_name: str,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]],
) -> str:
    component = component_library.get(component_name)
    if component is None:
        raise ValueError(f"component library does not define component '{component_name}'")
    supports = component.get("supports")
    if not isinstance(supports, dict):
        raise ValueError(f"component '{component_name}' must define object-valued 'supports'")

    node_opcode = {node.id: node.opcode for region in design.regions for node in region.nodes}
    state_to_opcode: list[tuple[str, int]] = []
    for controller in controllers:
        for emit in controller.emits:
            for issue in _iter_issue_actions(emit.attributes):
                instance, _, node_id = issue.partition("<-")
                if instance != resource_id or not node_id:
                    continue
                opcode_name = node_opcode.get(_issued_node_base_id(node_id))
                if opcode_name is None:
                    continue
                support = supports.get(opcode_name)
                if not isinstance(support, dict):
                    raise ValueError(f"component '{component_name}' does not support opcode '{opcode_name}'")
                opcode_literal = support.get("opcode")
                if not isinstance(opcode_literal, int):
                    raise ValueError(f"component '{component_name}' support '{opcode_name}' must define integer 'opcode'")
                condition = _state_eq_expr(_controller_state_id(controller, top_controller), controller_codes[controller.name][emit.state])
                state_to_opcode.append((condition, opcode_literal))

    if not state_to_opcode:
        return "0"
    unique_opcodes = {opcode for _, opcode in state_to_opcode}
    if len(unique_opcodes) == 1:
        return str(next(iter(unique_opcodes)))
    branches = [f"{condition} ? {opcode}" for condition, opcode in state_to_opcode]
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


def _issued_node_base_id(node_id: str) -> str:
    return node_id.split("@", 1)[0]


def _issued_node_occurrence_index(node_id: str) -> int | None:
    if "@" not in node_id:
        return None
    _, occurrence_text = node_id.split("@", 1)
    if not occurrence_text.isdigit():
        return None
    return int(occurrence_text)
