"""FSM-to-uglir lowering entrypoint."""

from __future__ import annotations

from math import ceil, log2
import re
from typing import Any

from uhls.backend.hls.lib import (
    format_component_spec,
    materialize_hdl_component_spec,
    parse_component_spec,
    resolve_component_type,
    resolve_component_definition,
)
from uhls.backend.hls.uhir.model import UHIRDesign
from .model import (
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


def lower_fsm_to_uglir(design: UHIRDesign, component_library: dict[str, dict[str, Any]] | None = None) -> UGLIRDesign:
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
    lowered = UGLIRDesign(name=design.name, component_libraries=list(design.component_libraries))
    helper_signal_ids: dict[tuple[str, str], str] = {}
    held_input_updates: list[tuple[str, str, str, str]] = []
    memory_interfaces = _memory_interfaces(design, component_library)
    phi_carries = _phi_carry_specs(design)
    _validate_memory_interface_schedule(top_controller, memory_interfaces)
    lowered.inputs = [
        UGLIRPort("input", "clk", "clock"),
        UGLIRPort("input", "rst", "i1"),
        UGLIRPort("input", "req_valid", "i1"),
        UGLIRPort("input", "resp_ready", "i1"),
    ]
    lowered.inputs.extend(_lowered_data_inputs(design, memory_interfaces))
    lowered.outputs = [
        UGLIRPort("output", "req_ready", "i1"),
        UGLIRPort("output", "resp_valid", "i1"),
    ]
    lowered.outputs.extend(_lowered_data_outputs(design, memory_interfaces))
    lowered.constants = list(design.constants)
    for interface in memory_interfaces.values():
        if interface["depth"] is not None:
            lowered.constants.append(UGLIRConstant(f"{interface['memory_name']}_depth", int(interface["depth"]), "u32"))

    lowered.resources.append(UGLIRResource("net", "req_fire", "i1"))
    lowered.resources.append(UGLIRResource("net", "resp_fire", "i1"))
    if component_library is not None and _needs_active_low_reset_helper(design, component_library):
        lowered.resources.append(UGLIRResource("net", "rst_n", "i1"))
    for interface in memory_interfaces.values():
        lowered.resources.append(UGLIRResource("port", interface["memory_name"], interface["component_name"], interface["memory_name"]))
    for controller in design.controllers:
        state_type = _state_type(controller)
        lowered.resources.append(UGLIRResource("reg", _controller_state_id(controller, top_controller), state_type))
        lowered.resources.append(UGLIRResource("net", _controller_next_state_id(controller, top_controller), state_type))
        for port in controller.inputs:
            signal_id = _controller_port_signal_id(controller, port.name, top_controller)
            if signal_id not in {"req_valid", "resp_ready"}:
                lowered.resources.append(UGLIRResource("net", signal_id, port.type))
        for port in controller.outputs:
            signal_id = _controller_port_signal_id(controller, port.name, top_controller)
            if signal_id not in {"req_ready", "resp_valid"}:
                lowered.resources.append(UGLIRResource("net", signal_id, port.type))
    for signal_name in _link_export_signal_names(design, top_controller):
        lowered.resources.append(UGLIRResource("net", signal_name, "i1"))

    for resource in design.resources:
        if resource.kind == "fu":
            component_kind = None if component_library is None else _component_kind(component_library, resource.value)
            if component_kind != "memory":
                instance_value = resource.value
                if component_library is not None:
                    instance_value = _materialized_instance_spec(component_library, resource.value)
                lowered.resources.append(UGLIRResource("inst", resource.id, instance_value, resource.value))
            if component_library is None:
                lowered.resources.append(UGLIRResource("net", f"{resource.id}_go", "i1"))
                result_type = _instance_result_type(design, resource.id)
                if result_type is not None:
                    lowered.resources.append(UGLIRResource("net", f"{resource.id}_y", result_type))
            else:
                should_hold_inputs = _component_should_hold_inputs(component_kind)
                held_input_names = {
                    port_name
                    for port_name, _ in _component_routed_input_ports(component_library, resource.value)
                    if should_hold_inputs
                }
                for port_name, port_type in _instance_ports(component_library, resource.value):
                    if _is_semantic_component_port_type(port_type):
                        continue
                    kind = "reg" if port_name in held_input_names else "net"
                    lowered.resources.append(UGLIRResource(kind, f"{resource.id}_{port_name}", port_type))
        elif resource.kind == "reg":
            lowered.resources.append(UGLIRResource("reg", resource.id, resource.value))
    _add_semantic_value_result_nets(lowered, design, component_library)
    for carry in phi_carries.values():
        lowered.resources.append(UGLIRResource("reg", carry["register"], carry["type"]))

    latch_targets = sorted(
        {
            register
            for controller in design.controllers
            for emit in controller.emits
            for register in emit.attributes.get("latch", ())
        }
    )
    for register in latch_targets:
        lowered.resources.append(UGLIRResource("net", f"latch_{register}", "i1"))
        lowered.resources.append(UGLIRResource("net", f"sel_{register}", "ctrl"))
        lowered.resources.append(UGLIRResource("mux", f"mx_{register}", _resource_value(design, register)))

    lowered.assigns.extend(
        [
            UGLIRAssign("req_fire", "req_valid & req_ready"),
            UGLIRAssign("resp_fire", "resp_valid & resp_ready"),
        ]
    )
    if any(resource.id == "rst_n" for resource in lowered.resources):
        lowered.assigns.append(UGLIRAssign("rst_n", "!rst"))
    for controller in design.controllers:
        for port in controller.outputs:
            lowered.assigns.append(
                UGLIRAssign(
                    _controller_port_signal_id(controller, port.name, top_controller),
                    _controller_output_expr(controller, port.name, controller_codes[controller.name], top_controller),
                )
            )
        lowered.assigns.append(
            UGLIRAssign(
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
            lowered.assigns.append(UGLIRAssign(f"{resource.id}_go", _issue_expr(design.controllers, top_controller, resource.id, controller_codes)))
            lowered.attachments.append(UGLIRAttach(resource.id, "go", f"{resource.id}_go"))
            if any(candidate.id == f"{resource.id}_y" for candidate in lowered.resources):
                lowered.attachments.append(UGLIRAttach(resource.id, "y", f"{resource.id}_y"))
            continue

        if _component_kind(component_library, resource.value) == "memory":
            for port_name, tie_expr in _component_tied_input_ports(component_library, resource.value).items():
                lowered.assigns.append(UGLIRAssign(f"{resource.id}_{port_name}", tie_expr))
            for port_name, port_type in _component_routed_input_ports(component_library, resource.value):
                mux_name = _lower_explicit_input_mux(
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
                    latched=_component_should_hold_inputs(_component_kind(component_library, resource.value)),
                )
                if _component_should_hold_inputs(_component_kind(component_library, resource.value)):
                    held_input_updates.append(
                        (
                            f"{resource.id}_{port_name}",
                            mux_name,
                            _issue_expr(design.controllers, top_controller, resource.id, controller_codes),
                            _default_net_expr(port_type),
                        )
                    )
            continue

        should_hold_inputs = _component_should_hold_inputs(_component_kind(component_library, resource.value))
        port_names = _instance_port_names(component_library, resource.value)
        issue_bindings = _component_issue_bindings(component_library, resource.value)
        for port_name, binding_key in issue_bindings.items():
            if port_name not in port_names:
                raise ValueError(
                    f"component '{resource.value}' issue binding references unknown port '{port_name}'"
                )
            lowered.assigns.append(
                UGLIRAssign(
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
        for port_name, tie_expr in _component_tied_input_ports(component_library, resource.value).items():
            lowered.assigns.append(UGLIRAssign(f"{resource.id}_{port_name}", tie_expr))
        if "op" in {
            port_name
            for port_name, _ in _component_routed_input_ports(component_library, resource.value)
        }:
            mux_name = _lower_opcode_input_mux(
                lowered,
                design,
                top_controller,
                controller_codes,
                component_library,
                helper_signal_ids,
                resource.id,
                resource.value,
                latched=should_hold_inputs,
            )
            if should_hold_inputs:
                held_input_updates.append(
                    (
                        f"{resource.id}_op",
                        mux_name,
                        _issue_expr(design.controllers, top_controller, resource.id, controller_codes),
                        _default_net_expr(_port_type(component_library, resource.value, "op")),
                    )
                )
        for port_name, port_type in _component_routed_input_ports(component_library, resource.value):
            if port_name == "op":
                continue
            mux_name = _lower_explicit_input_mux(
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
                latched=should_hold_inputs,
            )
            if should_hold_inputs:
                held_input_updates.append(
                    (
                        f"{resource.id}_{port_name}",
                        mux_name,
                        _issue_expr(design.controllers, top_controller, resource.id, controller_codes),
                        _default_net_expr(port_type),
                    )
                )
        for port_name in port_names:
            semantic_signal = _semantic_component_port_signal(component_library, resource.value, port_name)
            if semantic_signal is not None:
                lowered.attachments.append(UGLIRAttach(resource.id, port_name, semantic_signal))
                continue
            lowered.attachments.append(UGLIRAttach(resource.id, port_name, f"{resource.id}_{port_name}"))

    for register in latch_targets:
        lowered.assigns.append(
            UGLIRAssign(
                f"latch_{register}",
                _latch_expr(design, design.controllers, top_controller, register, controller_codes, component_library),
            )
        )
        lowered.assigns.append(
            UGLIRAssign(
                f"sel_{register}",
                _select_expr(design, design.controllers, top_controller, register, controller_codes, component_library),
            )
        )
        lowered.muxes.append(_build_register_mux(design, register, component_library))

    for assign in _memory_interface_assigns(memory_interfaces):
        lowered.assigns.append(assign)

    for output_name, driver in _output_drivers(design, component_library).items():
        lowered.assigns.append(UGLIRAssign(output_name, driver))

    top_seq_block = UGLIRSeqBlock(
        clock="clk",
        reset="rst",
        reset_updates=[UGLIRSeqUpdate(_controller_state_id(top_controller, top_controller), str(controller_codes[top_controller.name]["IDLE"]))],
        updates=[UGLIRSeqUpdate(_controller_state_id(top_controller, top_controller), _controller_next_state_id(top_controller, top_controller))],
    )
    for carry_source_id, carry in phi_carries.items():
        operands = tuple(carry["operands"])
        incoming = tuple(carry["incoming"])
        entry_like = bool(incoming) and incoming[0] == "entry"
        predecessor_update = _phi_carry_needs_predecessor_update(design, carry_source_id)
        init_region = _producer_region(design, operands[0])
        init_expr = _resolve_phi_carry_update_source(
            design,
            operands[0],
            0,
            component_library,
            region=init_region,
        )
        if init_expr is None:
            init_expr = _resolve_value_signal(design, operands[0], component_library, 0)

        update_sources: list[tuple[str, str]] = []
        start_index = 1 if entry_like else 0
        for operand in operands[start_index:]:
            operand_region = _producer_region(design, operand)
            if entry_like and operand in phi_carries:
                completion_steps = _loop_body_completion_steps(design, operand_region)
                if completion_steps:
                    for update_step in completion_steps:
                        state_name = f"T{update_step}"
                        if state_name not in controller_codes[top_controller.name]:
                            continue
                        source_expr = _resolve_phi_carry_update_source(
                            design,
                            operand,
                            update_step,
                            component_library,
                            region=operand_region,
                        )
                        if source_expr is None:
                            source_expr = _resolve_value_signal(
                                design,
                                operand,
                                component_library,
                                consumer_start=update_step,
                                region=operand_region,
                            )
                        update_sources.append(
                            (
                                _state_eq_expr(
                                    _controller_state_id(top_controller, top_controller),
                                    controller_codes[top_controller.name][state_name],
                                ),
                                source_expr,
                            )
                        )
                    continue
            operand_global_steps = sorted(_value_global_live_starts(design, operand))
            if not operand_global_steps:
                continue
            local_live_starts = _value_live_starts(design, operand, region=operand_region) if operand_region is not None else set()
            for step in operand_global_steps:
                producer_register = _register_for_value_at_step(design, operand, step, region=operand_region)
                if component_library is None or predecessor_update:
                    update_step = max(step - 1, 0)
                    source_step = update_step if component_library is not None else step
                else:
                    update_step = step
                    source_step = step
                state_name = f"T{update_step}"
                if state_name not in controller_codes[top_controller.name]:
                    continue
                source_expr = _resolve_phi_carry_update_source(
                    design,
                    operand,
                    source_step,
                    component_library,
                    region=operand_region,
                )
                if source_expr is None and producer_register is not None and source_step not in local_live_starts:
                    source_expr = producer_register
                if source_expr is None:
                    source_expr = _resolve_value_signal(
                        design,
                        operand,
                        component_library,
                        consumer_start=source_step,
                        region=operand_region,
                    )
                update_sources.append(
                    (
                        _state_eq_expr(
                            _controller_state_id(top_controller, top_controller),
                            controller_codes[top_controller.name][state_name],
                        ),
                        source_expr,
                    )
                )

        enable_parts: list[str] = ["req_fire"] if entry_like else []
        enable_parts.extend(condition for condition, _source in update_sources)
        enable_expr = " | ".join(enable_parts) if enable_parts else "false"

        if entry_like:
            value_expr = f"(req_fire) ? {init_expr}"
            if update_sources:
                value_expr += " : " + " : ".join(
                    f"({condition}) ? {source}"
                    for condition, source in update_sources
                )
                value_expr += f" : {carry['register']}"
            else:
                value_expr += f" : {carry['register']}"
            reset_expr = init_expr
        else:
            if update_sources:
                value_expr = " : ".join(
                    f"({condition}) ? {source}"
                    for condition, source in update_sources
                )
                value_expr += f" : {carry['register']}"
            else:
                value_expr = carry["register"]
            reset_expr = _default_net_expr(carry["type"])

        top_seq_block.reset_updates.append(UGLIRSeqUpdate(carry["register"], reset_expr))
        top_seq_block.updates.append(UGLIRSeqUpdate(carry["register"], value_expr, enable_expr))
    for target, value, enable, reset_value in held_input_updates:
        top_seq_block.reset_updates.append(UGLIRSeqUpdate(target, reset_value))
        top_seq_block.updates.append(UGLIRSeqUpdate(target, value, enable))
    for register in latch_targets:
        top_seq_block.updates.append(UGLIRSeqUpdate(register, f"mx_{register}", f"latch_{register}"))
    lowered.seq_blocks.append(top_seq_block)
    for controller in design.controllers:
        if controller.name == top_controller.name:
            continue
        lowered.seq_blocks.append(
            UGLIRSeqBlock(
                clock="clk",
                reset="rst",
                reset_updates=[UGLIRSeqUpdate(_controller_state_id(controller, top_controller), str(controller_codes[controller.name]["IDLE"]))],
                updates=[UGLIRSeqUpdate(_controller_state_id(controller, top_controller), _controller_next_state_id(controller, top_controller))],
            )
        )
    return _apply_signal_naming_convention(lowered)


def _apply_signal_naming_convention(design: UGLIRDesign) -> UGLIRDesign:
    rename_map = {
        resource.id: _uglir_signal_name(resource.kind, resource.id)
        for resource in design.resources
        if resource.kind in {"reg", "net", "mux"}
    }
    if not rename_map:
        return design

    normalized = UGLIRDesign(
        name=design.name,
        stage=design.stage,
        component_libraries=list(getattr(design, "component_libraries", ())),
    )
    normalized.inputs = list(design.inputs)
    normalized.outputs = list(design.outputs)
    normalized.constants = list(design.constants)

    for resource in design.resources:
        normalized.resources.append(
            UGLIRResource(
                resource.kind,
                rename_map.get(resource.id, resource.id),
                resource.value,
                resource.target,
            )
        )
    for assign in design.assigns:
        normalized.assigns.append(
            UGLIRAssign(
                rename_map.get(assign.target, assign.target),
                _rewrite_signal_expr(assign.expr, rename_map),
            )
        )
    for attachment in design.attachments:
        normalized.attachments.append(
            UGLIRAttach(
                attachment.instance,
                attachment.port,
                rename_map.get(attachment.signal, attachment.signal),
            )
        )
    for mux in design.muxes:
        normalized.muxes.append(
            UGLIRMux(
                name=rename_map.get(mux.name, mux.name),
                type=mux.type,
                select=rename_map.get(mux.select, mux.select),
                cases=[
                    UGLIRMuxCase(case.key, rename_map.get(case.source, case.source))
                    for case in mux.cases
                ],
            )
        )
    for seq_block in design.seq_blocks:
        normalized.seq_blocks.append(
            UGLIRSeqBlock(
                clock=seq_block.clock,
                reset=None if seq_block.reset is None else _rewrite_signal_expr(seq_block.reset, rename_map),
                reset_updates=[
                    UGLIRSeqUpdate(
                        rename_map.get(update.target, update.target),
                        _rewrite_signal_expr(update.value, rename_map),
                        None if update.enable is None else _rewrite_signal_expr(update.enable, rename_map),
                    )
                    for update in seq_block.reset_updates
                ],
                updates=[
                    UGLIRSeqUpdate(
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


def _latch_expr(
    design: UHIRDesign,
    controllers,
    top_controller,
    register: str,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]] | None = None,
) -> str:
    active_states = [condition for condition, _source in _register_state_sources(
        design,
        controllers,
        top_controller,
        register,
        controller_codes,
        component_library,
    )]
    if not active_states:
        return "false"
    ordered_conditions = list(dict.fromkeys(active_states))
    return " | ".join(ordered_conditions)


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
    branches = [
        f"{condition} ? {'HOLD' if source == register else labels[source]}"
        for condition, source in choices
    ]
    return " : ".join(branches) + " : HOLD"


def _build_register_mux(
    design: UHIRDesign,
    register: str,
    component_library: dict[str, dict[str, Any]] | None,
) -> UGLIRMux:
    register_type = _resource_value(design, register)
    mux = UGLIRMux(name=f"mx_{register}", type=register_type, select=f"sel_{register}")
    sources = [register, *_register_possible_sources(design, register, component_library)]
    labels = _register_mux_case_labels(register, sources)
    mux.cases.append(UGLIRMuxCase("HOLD", register))
    seen_sources = {register}
    for source in _register_possible_sources(design, register, component_library):
        if source in seen_sources:
            continue
        mux.cases.append(UGLIRMuxCase(labels[source], source))
        seen_sources.add(source)
    return mux


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
    child_refs_by_region: dict[str, list[str]] = {}
    region_by_id = {region.id: region for region in design.regions}
    for candidate in design.regions:
        refs: list[str] = []
        for node in candidate.nodes:
            for child_id in _node_children(node):
                if child_id in region_by_id:
                    refs.append(child_id)
        child_refs_by_region[candidate.id] = refs
    ordered: list[str] = []
    seen: set[str] = set()

    def visit(region_id: str) -> None:
        if region_id in seen:
            return
        seen.add(region_id)
        ordered.append(region_id)
        for child_id in children_by_parent.get(region_id, ()):
            visit(child_id)
        for child_id in child_refs_by_region.get(region_id, ()):
            visit(child_id)

    for root in roots:
        visit(root)
    return tuple(ordered)


def _node_children(node) -> tuple[str, ...]:
    children: list[str] = []
    for attr_name in ("child", "true_child", "false_child"):
        child = node.attributes.get(attr_name)
        if isinstance(child, str) and child:
            children.append(child)
    return tuple(children)


def _root_region_ids(design: UHIRDesign) -> tuple[str, ...]:
    referenced = {
        child_id
        for region in design.regions
        for node in region.nodes
        for child_id in _node_children(node)
    }
    return tuple(
        sorted(
            region.id
            for region in design.regions
            if region.parent is None and region.id not in referenced
        )
    )


def _child_region_shift(node, key: str) -> int:
    if key != "child":
        return 0
    node_start = node.attributes.get("start")
    if isinstance(node_start, int):
        return node_start
    return 0


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


def _controller_link_assigns(design: UHIRDesign, top_controller, controller_codes: dict[str, dict[str, int]]) -> list[UGLIRAssign]:
    controllers_by_name = {controller.name: controller for controller in design.controllers}
    assigns: list[UGLIRAssign] = []
    for parent in design.controllers:
        for link in parent.links:
            child = controllers_by_name.get(link.child)
            if child is None:
                raise ValueError(f"uglir lowering references unknown child controller '{link.child}'")
            act_mapping = link.attributes.get("act")
            if isinstance(act_mapping, tuple) and len(act_mapping) == 2:
                _, child_input = act_mapping
                assigns.append(
                    UGLIRAssign(
                        _controller_port_signal_id(child, child_input, top_controller),
                        _controller_action_expr(parent, top_controller, "activate", link.node, controller_codes[parent.name]),
                    )
                )
            ready_mapping = link.attributes.get("ready")
            if isinstance(ready_mapping, tuple) and len(ready_mapping) == 2:
                parent_signal, child_output = ready_mapping
                assigns.append(
                    UGLIRAssign(
                        str(parent_signal),
                        _controller_port_signal_id(child, str(child_output), top_controller),
                    )
                )
            done_mapping = link.attributes.get("done")
            if isinstance(done_mapping, tuple) and len(done_mapping) == 2:
                parent_signal, child_output = done_mapping
                assigns.append(
                    UGLIRAssign(
                        str(parent_signal),
                        _controller_port_signal_id(child, str(child_output), top_controller),
                    )
                )
            done_ready_mapping = link.attributes.get("done_ready")
            if isinstance(done_ready_mapping, tuple) and len(done_ready_mapping) == 2:
                parent_signal, child_input = done_ready_mapping
                assigns.append(
                    UGLIRAssign(
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
            if component_library is None:
                source = _resolve_producer_signal(design, binding.producer, component_library, region=region)
                if source is not None:
                    sources.append(source)
                continue
            capture_steps = sorted(_binding_global_capture_steps(design, region.id, binding, component_library))
            if not capture_steps:
                source = _resolve_producer_signal(design, binding.producer, component_library, region=region)
                if source is not None:
                    sources.append(source)
                continue
            for capture_step in capture_steps:
                source = _resolve_producer_signal(
                    design,
                    binding.producer,
                    component_library,
                    consumer_start=capture_step,
                    region=region,
                )
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
    choices_by_condition: dict[str, str] = {}
    for controller in controllers:
        if not controller.attributes.get("region"):
            continue
        for region in design.regions:
            for binding in region.value_bindings:
                if binding.register != register:
                    continue
                if component_library is None:
                    for capture_step in sorted(_binding_global_capture_steps(design, region.id, binding, component_library)):
                        source = _resolve_producer_signal(design, binding.producer, component_library, region=region)
                        if source is None:
                            continue
                        state_name = f"T{capture_step}"
                        if state_name not in controller_codes[controller.name]:
                            continue
                        condition = _state_eq_expr(
                            _controller_state_id(controller, top_controller),
                            controller_codes[controller.name][state_name],
                        )
                        choices_by_condition.setdefault(condition, source)
                    continue
                for capture_step in sorted(_binding_global_capture_steps(design, region.id, binding, component_library)):
                    source = _resolve_producer_signal(
                        design,
                        binding.producer,
                        component_library,
                        consumer_start=capture_step,
                        region=region,
                    )
                    if source is None:
                        continue
                    state_name = f"T{capture_step}"
                    if state_name not in controller_codes[controller.name]:
                        continue
                    condition = _state_eq_expr(
                        _controller_state_id(controller, top_controller),
                        controller_codes[controller.name][state_name],
                    )
                    choices_by_condition.setdefault(condition, source)
    return list(choices_by_condition.items())


def _value_capture_step(
    design: UHIRDesign,
    value_id: str,
    live_start: int,
    component_library: dict[str, dict[str, Any]] | None,
) -> int:
    if component_library is None:
        return live_start
    producer_node = _producer_node_map(design).get(value_id)
    if producer_node is None:
        return live_start
    bind = producer_node.attributes.get("bind")
    if not isinstance(bind, str):
        return live_start
    try:
        component_name = _resource_value(design, bind)
    except ValueError:
        return live_start
    component_kind = _component_kind(component_library, component_name)
    if component_kind in {"pipelined", "sequential"}:
        return max(live_start - 1, 0)
    return live_start


def _producer_register_map(design: UHIRDesign) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for region in design.regions:
        for binding in region.value_bindings:
            mapping.setdefault(binding.producer, binding.register)
    return mapping


def _bindings_for_value(design: UHIRDesign, value_id: str, *, region=None) -> tuple[object | None, tuple[object, ...]]:
    producer_region = region if region is not None and _region_producer_node(region, value_id) is not None else _producer_region(design, value_id)
    if producer_region is None:
        return None, ()
    producer_ids = {value_id}
    local_producer = _region_producer_node(producer_region, value_id)
    if local_producer is not None and local_producer.id != value_id:
        producer_ids.add(local_producer.id)
    bindings = tuple(binding for binding in producer_region.value_bindings if binding.producer in producer_ids)
    return producer_region, bindings


def _capture_registers_by_step(
    design: UHIRDesign,
    value_id: str,
    component_library: dict[str, dict[str, Any]] | None,
    *,
    region=None,
) -> dict[int, set[str]]:
    producer_region, bindings = _bindings_for_value(design, value_id, region=region)
    if producer_region is None:
        return {}
    capture_registers: dict[int, set[str]] = {}
    for binding in bindings:
        for capture_step in _binding_global_capture_steps(
            design,
            producer_region.id,
            binding,
            component_library,
        ):
            capture_registers.setdefault(capture_step, set()).add(binding.register)
    return capture_registers


def _register_for_value_at_step(
    design: UHIRDesign,
    value_id: str,
    consumer_start: int | None,
    *,
    occurrence_index: int | None = None,
    region=None,
) -> str | None:
    _producer_region_obj, bindings = _bindings_for_value(design, value_id, region=region)
    if not bindings:
        return None
    if isinstance(occurrence_index, int) and occurrence_index >= 0:
        ordered_intervals = sorted(
            (live_start, live_end, binding.register)
            for binding in bindings
            for live_start, live_end in binding.live_intervals
        )
        if occurrence_index < len(ordered_intervals):
            _live_start, _live_end, register = ordered_intervals[occurrence_index]
            return register
    if isinstance(consumer_start, int):
        probe_steps = (max(consumer_start - 1, 0), consumer_start)
        for probe_step in probe_steps:
            for binding in bindings:
                for live_start, live_end in binding.live_intervals:
                    if live_start <= probe_step <= live_end:
                        return binding.register
        for binding in bindings:
            for live_start, _live_end in binding.live_intervals:
                if live_start == consumer_start:
                    return binding.register
    unique_registers = tuple(dict.fromkeys(binding.register for binding in bindings))
    if len(unique_registers) == 1:
        return unique_registers[0]
    return bindings[0].register


def _phi_carry_specs(design: UHIRDesign) -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    for region in design.regions:
        source_by_node = {
            source_map.node_id: source_map.source_id
            for source_map in region.mappings
        }
        for node in region.nodes:
            if node.opcode != "phi" or len(node.operands) < 2:
                continue
            source_id = source_by_node.get(node.id)
            if not isinstance(source_id, str) or not source_id:
                continue
            if not isinstance(node.result_type, str) or not node.result_type:
                continue
            loop_role = node.attributes.get("loop_role")
            if loop_role not in {"header", "body"}:
                continue
            specs[source_id] = {
                "register": _phi_carry_register_id(source_id),
                "type": node.result_type,
                "operands": tuple(node.operands),
                "incoming": tuple(node.attributes.get("incoming", ())),
            }
    return specs


def _phi_carry_register_id(source_id: str) -> str:
    return f"phi_{source_id}"


def _phi_carry_needs_predecessor_update(design: UHIRDesign, source_id: str) -> bool:
    producer_region = _producer_region(design, source_id)
    producer_node = _region_producer_node(producer_region, source_id)
    if producer_node is None or producer_node.opcode != "phi":
        return False
    loop_member = producer_node.attributes.get("loop_member")
    for region in design.regions:
        if region.id == getattr(producer_region, "id", None):
            continue
        for node in region.nodes:
            if source_id not in getattr(node, "operands", ()):
                continue
            if node.attributes.get("loop_member") != loop_member:
                continue
            if node.attributes.get("loop_role") != "body":
                continue
            if node.attributes.get("start") == 0:
                return True
    return False


def _loop_body_completion_steps(design: UHIRDesign, body_region) -> tuple[int, ...]:
    if body_region is None or getattr(body_region, "kind", None) != "body":
        return ()
    parent_id = getattr(body_region, "parent", None)
    if not isinstance(parent_id, str) or not parent_id:
        return ()
    body_end_steps = [
        node.attributes.get("end")
        for node in body_region.nodes
        if isinstance(node.attributes.get("end"), int)
    ]
    if not body_end_steps:
        return ()
    body_end = max(body_end_steps)
    completion_steps: list[int] = []
    for region in design.regions:
        for node in region.nodes:
            if node.opcode != "loop" or node.attributes.get("child") != parent_id:
                continue
            loop_start = node.attributes.get("start", 0)
            trip_count = node.attributes.get("static_trip_count")
            iter_ii = node.attributes.get("iter_initiation_interval")
            if not isinstance(loop_start, int) or not isinstance(trip_count, int) or not isinstance(iter_ii, int):
                continue
            completion_steps.extend(loop_start + iteration * iter_ii + body_end for iteration in range(trip_count))
    return tuple(sorted(set(completion_steps)))


def _value_live_starts(design: UHIRDesign, value_id: str, region=None) -> set[int]:
    live_starts: set[int] = set()
    regions = design.regions if region is None else [region]
    for candidate_region in regions:
        for binding in candidate_region.value_bindings:
            if binding.producer == value_id:
                for live_start, _live_end in binding.live_intervals:
                    live_starts.add(live_start)
    if live_starts:
        return live_starts
    producer_node = _region_producer_node(region, value_id)
    if producer_node is not None and producer_node.opcode == "phi":
        incoming = tuple(producer_node.attributes.get("incoming", ()))
        if incoming and incoming[0] != "entry":
            operand_steps: set[int] = set()
            for operand in producer_node.operands:
                operand_region = _producer_region(design, operand)
                operand_steps.update(_value_live_starts(design, operand, region=operand_region))
            if operand_steps:
                return operand_steps
        start = producer_node.attributes.get("start")
        if isinstance(start, int):
            return {start}
    if producer_node is not None and producer_node.id != value_id:
        return _value_live_starts(design, producer_node.id, region=region)
    if producer_node is None:
        producer_node = _producer_node_map(design).get(value_id)
    if producer_node is None or getattr(producer_node, "opcode", None) != "call":
        return live_starts
    child_region = design.get_region(producer_node.attributes.get("child"))
    returned_value = _call_return_value_id(design, producer_node)
    if returned_value is None or returned_value == value_id:
        return live_starts
    return _value_live_starts(design, returned_value, region=child_region)
    return live_starts


def _value_global_live_starts(design: UHIRDesign, value_id: str) -> set[int]:
    live_starts: set[int] = set()
    region_by_id = {region.id: region for region in design.regions}

    def append_region(region_id: str, offset: int) -> None:
        region = region_by_id[region_id]
        if _region_producer_node(region, value_id) is not None:
            live_starts.update(offset + step for step in _value_live_starts(design, value_id, region=region))
        for node in region.nodes:
            node_start = node.attributes.get("start")
            if not isinstance(node_start, int):
                continue
            if node.opcode == "loop":
                child_id = node.attributes.get("child")
                if design.stage == "fsm":
                    if isinstance(child_id, str):
                        append_loop_header(child_id, offset, expand_body=True)
                    continue
                trip_count = node.attributes.get("static_trip_count")
                iter_ii = node.attributes.get("iter_initiation_interval")
                if not isinstance(child_id, str) or not isinstance(trip_count, int) or not isinstance(iter_ii, int):
                    continue
                for iteration in range(trip_count):
                    append_loop_header(child_id, offset + node_start + iteration * iter_ii, expand_body=True)
                append_loop_header(child_id, offset + node_start + trip_count * iter_ii, expand_body=False)
                continue
            for key in ("child", "true_child", "false_child"):
                child_id = node.attributes.get(key)
                if not isinstance(child_id, str) or not child_id:
                    continue
                child_offset = offset if design.stage == "fsm" else offset + _child_region_shift(node, key)
                append_region(child_id, child_offset)

    def append_loop_header(region_id: str, offset: int, *, expand_body: bool) -> None:
        region = region_by_id[region_id]
        if _region_producer_node(region, value_id) is not None:
            live_starts.update(offset + step for step in _value_live_starts(design, value_id, region=region))
        for node in region.nodes:
            if node.opcode == "branch":
                true_child = node.attributes.get("true_child")
                false_child = node.attributes.get("false_child")
                if design.stage == "fsm":
                    if isinstance(true_child, str):
                        append_region(true_child, offset)
                    if isinstance(false_child, str):
                        append_region(false_child, offset)
                    continue
                if expand_body and isinstance(true_child, str):
                    append_region(true_child, offset)
                if not expand_body and isinstance(false_child, str):
                    append_region(false_child, offset)
                continue
            for key in ("child", "true_child", "false_child"):
                child_id = node.attributes.get(key)
                if not isinstance(child_id, str) or not child_id:
                    continue
                child_offset = offset if design.stage == "fsm" else offset + _child_region_shift(node, key)
                append_region(child_id, child_offset)

    for region_id in _root_region_ids(design):
        append_region(region_id, 0)
    return live_starts


def _producer_global_capture_steps(
    design: UHIRDesign,
    producer_region_id: str,
    producer_id: str,
    component_library: dict[str, dict[str, Any]] | None,
) -> set[int]:
    if design.stage == "fsm":
        region = design.get_region(producer_region_id)
        if region is None:
            return set()
        matched = False
        capture_steps: set[int] = set()
        for binding in region.value_bindings:
            if binding.producer != producer_id:
                continue
            matched = True
            for live_start, _live_end in binding.live_intervals:
                capture_steps.add(_value_capture_step(design, binding.producer, live_start, component_library))
        if matched:
            return capture_steps
        producer_node = _region_producer_node(region, producer_id)
        if producer_node is not None and producer_node.opcode == "phi":
            incoming = tuple(producer_node.attributes.get("incoming", ()))
            if incoming and incoming[0] != "entry":
                for operand in producer_node.operands:
                    operand_region = _producer_region(design, operand)
                    operand_steps = sorted(_value_live_starts(design, operand, region=operand_region))
                    for live_start in operand_steps:
                        capture_steps.add(_value_capture_step(design, operand, live_start, component_library))
                if capture_steps:
                    return capture_steps
            start = producer_node.attributes.get("start")
            if isinstance(start, int):
                return {start}
        return capture_steps

    capture_steps: set[int] = set()
    region_by_id = {region.id: region for region in design.regions}

    def append_region(region_id: str, offset: int) -> None:
        region = region_by_id[region_id]
        if region.id == producer_region_id:
            matched = False
            for binding in region.value_bindings:
                if binding.producer != producer_id:
                    continue
                matched = True
                for live_start, _live_end in binding.live_intervals:
                    capture_steps.add(offset + _value_capture_step(design, binding.producer, live_start, component_library))
            if not matched:
                producer_node = _region_producer_node(region, producer_id)
                if producer_node is not None and producer_node.opcode == "phi":
                    incoming = tuple(producer_node.attributes.get("incoming", ()))
                    used_operand_steps = False
                    if incoming and incoming[0] != "entry":
                        for operand in producer_node.operands:
                            operand_region = _producer_region(design, operand)
                            operand_steps = sorted(_value_live_starts(design, operand, region=operand_region))
                            for live_start in operand_steps:
                                capture_steps.add(offset + _value_capture_step(design, operand, live_start, component_library))
                        used_operand_steps = True
                    if not used_operand_steps:
                        start = producer_node.attributes.get("start")
                        if isinstance(start, int):
                            capture_steps.add(offset + start)
        for node in region.nodes:
            node_start = node.attributes.get("start")
            if not isinstance(node_start, int):
                continue
            if node.opcode == "loop":
                child_id = node.attributes.get("child")
                if design.stage == "fsm":
                    if isinstance(child_id, str):
                        append_loop_header(child_id, offset, expand_body=True)
                    continue
                trip_count = node.attributes.get("static_trip_count")
                iter_ii = node.attributes.get("iter_initiation_interval")
                if not isinstance(child_id, str) or not isinstance(trip_count, int) or not isinstance(iter_ii, int):
                    continue
                for iteration in range(trip_count):
                    append_loop_header(child_id, offset + node_start + iteration * iter_ii, expand_body=True)
                append_loop_header(child_id, offset + node_start + trip_count * iter_ii, expand_body=False)
                continue
            for key in ("child", "true_child", "false_child"):
                child_id = node.attributes.get(key)
                if not isinstance(child_id, str) or not child_id:
                    continue
                child_offset = offset if design.stage == "fsm" else offset + _child_region_shift(node, key)
                append_region(child_id, child_offset)

    def append_loop_header(region_id: str, offset: int, *, expand_body: bool) -> None:
        region = region_by_id[region_id]
        if region.id == producer_region_id:
            matched = False
            for binding in region.value_bindings:
                if binding.producer != producer_id:
                    continue
                matched = True
                for live_start, _live_end in binding.live_intervals:
                    capture_steps.add(offset + _value_capture_step(design, binding.producer, live_start, component_library))
            if not matched:
                producer_node = _region_producer_node(region, producer_id)
                if producer_node is not None and producer_node.opcode == "phi":
                    incoming = tuple(producer_node.attributes.get("incoming", ()))
                    used_operand_steps = False
                    if incoming and incoming[0] != "entry":
                        for operand in producer_node.operands:
                            operand_region = _producer_region(design, operand)
                            operand_steps = sorted(_value_live_starts(design, operand, region=operand_region))
                            for live_start in operand_steps:
                                capture_steps.add(offset + _value_capture_step(design, operand, live_start, component_library))
                        used_operand_steps = True
                    if not used_operand_steps:
                        start = producer_node.attributes.get("start")
                        if isinstance(start, int):
                            capture_steps.add(offset + start)
        for node in region.nodes:
            if node.opcode == "branch":
                true_child = node.attributes.get("true_child")
                false_child = node.attributes.get("false_child")
                if design.stage == "fsm":
                    if isinstance(true_child, str):
                        append_region(true_child, offset)
                    if isinstance(false_child, str):
                        append_region(false_child, offset)
                    continue
                if expand_body and isinstance(true_child, str):
                    append_region(true_child, offset)
                if not expand_body and isinstance(false_child, str):
                    append_region(false_child, offset)
                continue
            for key in ("child", "true_child", "false_child"):
                child_id = node.attributes.get(key)
                if not isinstance(child_id, str) or not child_id:
                    continue
                child_offset = offset if design.stage == "fsm" else offset + _child_region_shift(node, key)
                append_region(child_id, child_offset)

    for region_id in _root_region_ids(design):
        append_region(region_id, 0)
    return capture_steps


def _binding_global_capture_steps(
    design: UHIRDesign,
    producer_region_id: str,
    binding,
    component_library: dict[str, dict[str, Any]] | None,
) -> set[int]:
    if design.stage == "fsm":
        return {
            _value_capture_step(design, binding.producer, live_start, component_library)
            for live_start, _live_end in binding.live_intervals
        }

    capture_steps: set[int] = set()
    region_by_id = {region.id: region for region in design.regions}

    def append_region(region_id: str, offset: int) -> None:
        region = region_by_id[region_id]
        if region.id == producer_region_id:
            for live_start, _live_end in binding.live_intervals:
                capture_steps.add(offset + _value_capture_step(design, binding.producer, live_start, component_library))
        for node in region.nodes:
            node_start = node.attributes.get("start")
            if not isinstance(node_start, int):
                continue
            if node.opcode == "loop":
                child_id = node.attributes.get("child")
                if design.stage == "fsm":
                    if isinstance(child_id, str):
                        append_loop_header(child_id, offset + node_start, expand_body=True)
                    continue
                trip_count = node.attributes.get("static_trip_count")
                iter_ii = node.attributes.get("iter_initiation_interval")
                if not isinstance(child_id, str) or not isinstance(trip_count, int) or not isinstance(iter_ii, int):
                    continue
                for iteration in range(trip_count):
                    append_loop_header(child_id, offset + node_start + iteration * iter_ii, expand_body=True)
                append_loop_header(child_id, offset + node_start + trip_count * iter_ii, expand_body=False)
                continue
            for key in ("child", "true_child", "false_child"):
                child_id = node.attributes.get(key)
                if not isinstance(child_id, str) or not child_id:
                    continue
                append_region(child_id, offset + _child_region_shift(node, key))

    def append_loop_header(region_id: str, offset: int, *, expand_body: bool) -> None:
        region = region_by_id[region_id]
        if region.id == producer_region_id:
            for live_start, _live_end in binding.live_intervals:
                capture_steps.add(offset + _value_capture_step(design, binding.producer, live_start, component_library))
        for node in region.nodes:
            if node.opcode == "branch":
                true_child = node.attributes.get("true_child")
                false_child = node.attributes.get("false_child")
                if design.stage == "fsm":
                    if isinstance(true_child, str):
                        append_region(true_child, offset)
                    if isinstance(false_child, str):
                        append_region(false_child, offset)
                    continue
                if expand_body and isinstance(true_child, str):
                    append_region(true_child, offset)
                if not expand_body and isinstance(false_child, str):
                    append_region(false_child, offset)
                continue
            for key in ("child", "true_child", "false_child"):
                child_id = node.attributes.get(key)
                if not isinstance(child_id, str) or not child_id:
                    continue
                append_region(child_id, offset + _child_region_shift(node, key))

    for region_id in _root_region_ids(design):
        append_region(region_id, 0)
    return capture_steps


def _binding_global_live_intervals(
    design: UHIRDesign,
    producer_region_id: str,
    binding,
) -> tuple[tuple[int, int], ...]:
    intervals: list[tuple[int, int]] = []
    region_by_id = {region.id: region for region in design.regions}

    def append_region(region_id: str, offset: int) -> None:
        region = region_by_id[region_id]
        if region.id == producer_region_id:
            for live_start, live_end in binding.live_intervals:
                intervals.append((offset + live_start, offset + live_end))
        for node in region.nodes:
            node_start = node.attributes.get("start")
            if not isinstance(node_start, int):
                continue
            if node.opcode == "loop":
                child_id = node.attributes.get("child")
                if design.stage == "fsm":
                    if isinstance(child_id, str):
                        append_loop_header(child_id, offset + node_start, expand_body=True)
                    continue
                trip_count = node.attributes.get("static_trip_count")
                iter_ii = node.attributes.get("iter_initiation_interval")
                if not isinstance(child_id, str) or not isinstance(trip_count, int) or not isinstance(iter_ii, int):
                    continue
                for iteration in range(trip_count):
                    append_loop_header(child_id, offset + node_start + iteration * iter_ii, expand_body=True)
                append_loop_header(child_id, offset + node_start + trip_count * iter_ii, expand_body=False)
                continue
            for key in ("child", "true_child", "false_child"):
                child_id = node.attributes.get(key)
                if not isinstance(child_id, str) or not child_id:
                    continue
                append_region(child_id, offset + _child_region_shift(node, key))

    def append_loop_header(region_id: str, offset: int, *, expand_body: bool) -> None:
        region = region_by_id[region_id]
        if region.id == producer_region_id:
            for live_start, live_end in binding.live_intervals:
                intervals.append((offset + live_start, offset + live_end))
        for node in region.nodes:
            if node.opcode == "branch":
                true_child = node.attributes.get("true_child")
                false_child = node.attributes.get("false_child")
                if design.stage == "fsm":
                    if isinstance(true_child, str):
                        append_region(true_child, offset)
                    if isinstance(false_child, str):
                        append_region(false_child, offset)
                    continue
                if expand_body and isinstance(true_child, str):
                    append_region(true_child, offset)
                if not expand_body and isinstance(false_child, str):
                    append_region(false_child, offset)
                continue
            for key in ("child", "true_child", "false_child"):
                child_id = node.attributes.get(key)
                if not isinstance(child_id, str) or not child_id:
                    continue
                append_region(child_id, offset + _child_region_shift(node, key))

    for region_id in _root_region_ids(design):
        append_region(region_id, 0)
    return tuple(intervals)


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
    return _parse_memref_type(type_name) is not None


def _parse_memref_type(type_name: str) -> tuple[str, int | None] | None:
    match = re.fullmatch(r"memref<\s*([A-Za-z_][\w$<>]*)\s*(?:,\s*(\d+)\s*)?>", type_name)
    if match is None:
        return None
    element_type, extent_text = match.groups()
    return element_type, None if extent_text is None else int(extent_text)


def _component_definition(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
) -> tuple[str, dict[str, Any]]:
    base_name, _, component = resolve_component_definition(component_library, component_name)
    return base_name, component


def _component_params(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
) -> dict[str, str]:
    _base_name, params, _component = resolve_component_definition(component_library, component_name)
    return params


def _materialized_instance_spec(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
) -> str:
    return materialize_hdl_component_spec(component_library, component_name)


def _parse_component_spec(component_name: str) -> tuple[str, dict[str, str]]:
    return parse_component_spec(component_name)


def _format_component_spec(base_name: str, params: dict[str, str]) -> str:
    return format_component_spec(base_name, params)


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


def _component_kind(component_library: dict[str, dict[str, Any]], component_name: str) -> str | None:
    _, component = _component_definition(component_library, component_name)
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
    resolved_port_name = _memory_port_name(component_library, component_name, port_name)
    if resolved_port_name is None:
        return None
    for candidate_name, candidate_type in _instance_ports(component_library, component_name):
        if candidate_name == resolved_port_name:
            return candidate_type
    return None


def _memory_port_name(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
    port_name: str,
) -> str | None:
    for candidate_name, _candidate_type in _instance_ports(component_library, component_name):
        if candidate_name == port_name:
            return candidate_name
    semantic_role = {
        "addr": "operand1",
        "wdata": "operand2",
        "rdata": "result",
    }.get(port_name)
    if semantic_role is not None:
        for opcode_name in ("load", "store"):
            try:
                support_port = _component_support_port_for_binding(
                    component_library,
                    component_name,
                    opcode_name,
                    semantic_role,
                    require_direction="output" if port_name == "rdata" else "input",
                )
            except ValueError:
                continue
            if support_port is not None:
                return support_port
    if port_name == "we":
        for literal in ("true", "false"):
            try:
                support_port = _component_support_port_for_binding(
                    component_library,
                    component_name,
                    "store",
                    literal,
                    require_direction="input",
                )
            except ValueError:
                continue
            if support_port is not None:
                return support_port
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
    returned_values: list[tuple[str, int | None, object | None]] = []
    for region in design.regions:
        if region.parent is not None:
            continue
        for node in region.nodes:
            if node.opcode == "ret" and node.operands:
                start = node.attributes.get("start")
                returned_values.append((node.operands[0], start if isinstance(start, int) else None, region))
    drivers: dict[str, str] = {}
    for output, (returned, consumer_start, region) in zip(design.outputs, returned_values, strict=False):
        register = _register_for_value_at_step(design, returned, consumer_start, region=region)
        if register is not None:
            drivers[output.name] = register
        else:
            drivers[output.name] = _resolve_value_signal(
                design,
                returned,
                component_library,
                consumer_start=consumer_start,
                region=region,
            )
    return drivers


def _lowered_data_inputs(design: UHIRDesign, memory_interfaces: dict[str, dict[str, Any]]) -> list[UGLIRPort]:
    lowered: list[UGLIRPort] = []
    for port in design.inputs:
        if not _is_memref_type(port.type) or port.name not in memory_interfaces:
            lowered.append(UGLIRPort(port.direction, port.name, port.type))
            continue
        interface = memory_interfaces[port.name]
        if interface["read_type"] is not None:
            lowered.append(UGLIRPort("input", f"{port.name}_rdata", interface["read_type"]))
    return lowered


def _lowered_data_outputs(design: UHIRDesign, memory_interfaces: dict[str, dict[str, Any]]) -> list[UGLIRPort]:
    lowered: list[UGLIRPort] = []
    for port in design.outputs:
        if not _is_memref_type(port.type) or port.name not in memory_interfaces:
            lowered.append(UGLIRPort(port.direction, port.name, port.type))
    seen_memories: set[str] = set()
    for port in [*design.inputs, *design.outputs]:
        if not _is_memref_type(port.type) or port.name not in memory_interfaces or port.name in seen_memories:
            continue
        seen_memories.add(port.name)
        interface = memory_interfaces[port.name]
        if interface["addr_type"] is not None:
            lowered.append(UGLIRPort("output", f"{port.name}_addr", interface["addr_type"]))
        if interface["has_write"] and interface["write_type"] is not None:
            lowered.append(UGLIRPort("output", f"{port.name}_wdata", interface["write_type"]))
        if interface["has_write"]:
            lowered.append(UGLIRPort("output", f"{port.name}_we", "i1"))
    return lowered


def _memory_interface_assigns(memory_interfaces: dict[str, dict[str, Any]]) -> list[UGLIRAssign]:
    assigns: list[UGLIRAssign] = []
    for interface in memory_interfaces.values():
        memory_name = interface["memory_name"]
        instance_id = interface["instance_id"]
        if interface["addr_type"] is not None and interface["addr_port"] is not None:
            assigns.append(UGLIRAssign(f"{memory_name}_addr", f"{instance_id}_{interface['addr_port']}"))
        if interface["has_write"] and interface["write_type"] is not None and interface["write_port"] is not None:
            assigns.append(UGLIRAssign(f"{memory_name}_wdata", f"{instance_id}_{interface['write_port']}"))
        if interface["has_write"] and interface["we_port"] is not None:
            assigns.append(UGLIRAssign(f"{memory_name}_we", f"{instance_id}_{interface['we_port']}"))
        if interface["read_type"] is not None and interface["read_port"] is not None:
            assigns.append(UGLIRAssign(f"{instance_id}_{interface['read_port']}", f"{memory_name}_rdata"))
    return assigns


def _memory_interfaces(
    design: UHIRDesign,
    component_library: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if component_library is None:
        return {}

    top_level_memories = {
        port.name: _parse_memref_type(port.type)
        for port in [*design.inputs, *design.outputs]
        if _is_memref_type(port.type)
    }
    memory_port_specs = {
        (resource.target if resource.target is not None else resource.id): resource.value
        for resource in design.resources
        if resource.kind == "port"
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
            memref_spec = top_level_memories.get(memory_name)
            if memref_spec is None:
                raise ValueError(
                    f"memory-bound node '{node.id}' references memory '{memory_name}' that is not a top-level memref port"
                )
            component_name = instance_components[instance_id]
            existing = interfaces.get(memory_name)
            if existing is None:
                component_spec = memory_port_specs.get(memory_name)
                if component_spec is None:
                    component_spec = _parameterized_memory_component_name(component_name, memref_spec)
                else:
                    _validate_memory_component_spec(component_spec, memref_spec, memory_name)
                interfaces[memory_name] = {
                    "memory_name": memory_name,
                    "component_name": component_spec,
                    "instance_id": instance_id,
                    "has_read": False,
                    "has_write": False,
                    "addr_port": _memory_port_name(component_library, component_name, "addr"),
                    "addr_type": _memory_port_type(component_library, component_name, "addr"),
                    "write_port": _memory_port_name(component_library, component_name, "wdata"),
                    "write_type": _memory_port_type(component_library, component_name, "wdata"),
                    "we_port": _memory_port_name(component_library, component_name, "we"),
                    "read_port": _memory_port_name(component_library, component_name, "rdata"),
                    "read_type": _memory_port_type(component_library, component_name, "rdata"),
                    "depth": memref_spec[1],
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


def _parameterized_memory_component_name(component_name: str, memref_spec: tuple[str, int | None]) -> str:
    base_name, params = _parse_component_spec(component_name)
    params = dict(params)
    params.setdefault("word_t", memref_spec[0])
    if memref_spec[1] is not None:
        params.setdefault("word_len", str(memref_spec[1]))
    return _format_component_spec(base_name, params)


def _validate_memory_component_spec(
    component_spec: str,
    memref_spec: tuple[str, int | None],
    memory_name: str,
) -> None:
    _, params = _parse_component_spec(component_spec)
    expected_word_t, expected_word_len = memref_spec
    actual_word_t = params.get("word_t")
    if actual_word_t is not None and actual_word_t != expected_word_t:
        raise ValueError(
            f"memory port '{memory_name}' uses '{component_spec}' but top-level memref expects word_t={expected_word_t}"
        )
    actual_word_len = params.get("word_len")
    if expected_word_len is not None and actual_word_len is not None and actual_word_len != str(expected_word_len):
        raise ValueError(
            f"memory port '{memory_name}' uses '{component_spec}' but top-level memref expects word_len={expected_word_len}"
        )


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
    node_region_by_id = {node.id: region for region in design.regions for node in region.nodes}
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
                source_expr = _binding_key_expr(
                    design,
                    node_region_by_id.get(_issued_node_base_id(node_id)),
                    node,
                    binding_key,
                    component_library,
                    _issued_node_occurrence_index(node_id),
                )
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
    node_region_by_id = {node.id: region for region in design.regions for node in region.nodes}
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
                    node_region_by_id.get(_issued_node_base_id(node_id)),
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


def _opcode_port_choices(
    design: UHIRDesign,
    controllers,
    top_controller,
    resource_id: str,
    component_name: str,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    node_opcode = {
        node.id: node.opcode
        for region in design.regions
        for node in region.nodes
    }
    _, component = _component_definition(component_library, component_name)
    supports = component.get("supports")
    if not isinstance(supports, dict):
        raise ValueError(f"component '{component_name}' must define object-valued 'supports'")
    choices: list[tuple[str, str]] = []
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
                    continue
                opcode_literal = support.get("opcode")
                if not isinstance(opcode_literal, int):
                    continue
                condition = _state_eq_expr(
                    _controller_state_id(controller, top_controller),
                    controller_codes[controller.name][emit.state],
                )
                choices.append((condition, str(opcode_literal)))
    return choices


def _lower_explicit_input_mux(
    lowered: UGLIRDesign,
    design: UHIRDesign,
    top_controller,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]],
    helper_signal_ids: dict[tuple[str, str], str],
    resource_id: str,
    component_name: str,
    port_name: str,
    port_type: str,
    *,
    latched: bool,
) -> str:
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
    return _lower_input_choices_mux(
        lowered,
        helper_signal_ids,
        resource_id,
        port_name,
        port_type,
        operand_choices,
        latched=latched,
    )


def _lower_opcode_input_mux(
    lowered: UGLIRDesign,
    design: UHIRDesign,
    top_controller,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]],
    helper_signal_ids: dict[tuple[str, str], str],
    resource_id: str,
    component_name: str,
    *,
    latched: bool,
) -> str:
    port_type = _port_type(component_library, component_name, "op")
    opcode_choices = _opcode_port_choices(
        design,
        design.controllers,
        top_controller,
        resource_id,
        component_name,
        controller_codes,
        component_library,
    )
    return _lower_input_choices_mux(
        lowered,
        helper_signal_ids,
        resource_id,
        "op",
        port_type,
        opcode_choices,
        latched=latched,
    )


def _lower_input_choices_mux(
    lowered: UGLIRDesign,
    helper_signal_ids: dict[tuple[str, str], str],
    resource_id: str,
    port_name: str,
    port_type: str,
    input_choices: list[tuple[str, str]],
    *,
    latched: bool,
) -> str:
    target_signal = f"{resource_id}_{port_name}"
    select_signal = f"sel_{resource_id}_{port_name}"
    mux_name = f"mx_{resource_id}_{port_name}"

    lowered.resources.append(UGLIRResource("net", select_signal, "ctrl"))
    lowered.resources.append(UGLIRResource("mux", mux_name, port_type))

    default_signal = _materialize_glue_source_signal(
        lowered,
        _default_net_expr(port_type),
        port_type,
        helper_signal_ids,
    )
    choices = [
        (
            condition,
            source_expr,
            _materialize_glue_source_signal(lowered, source_expr, port_type, helper_signal_ids),
        )
        for condition, source_expr in input_choices
    ]
    labels = _input_mux_case_labels(_default_net_expr(port_type), default_signal, choices)

    if choices:
        select_expr = " : ".join(f"{condition} ? {labels[source_signal]}" for condition, _source_expr, source_signal in choices)
        select_expr = f"{select_expr} : {labels[default_signal]}"
    else:
        select_expr = labels[default_signal]
    lowered.assigns.append(UGLIRAssign(select_signal, select_expr))
    if not latched:
        lowered.assigns.append(UGLIRAssign(target_signal, mux_name))

    mux = UGLIRMux(name=mux_name, type=port_type, select=select_signal)
    seen_sources: set[str] = set()
    for source_signal in [default_signal, *(source for _, _expr, source in choices)]:
        if source_signal in seen_sources:
            continue
        seen_sources.add(source_signal)
        mux.cases.append(UGLIRMuxCase(labels[source_signal], source_signal))
    lowered.muxes.append(mux)
    return mux_name


def _materialize_glue_source_signal(
    lowered: UGLIRDesign,
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
    lowered.resources.append(UGLIRResource("net", signal_id, port_type))
    lowered.assigns.append(UGLIRAssign(signal_id, source_expr))
    helper_signal_ids[key] = signal_id
    return signal_id


def _is_known_uglir_signal(lowered: UGLIRDesign, signal_name: str) -> bool:
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
    _, component = _component_definition(component_library, component_name)
    params = _component_params(component_library, component_name)
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
        normalized.append((str(port_name), resolve_component_type(port_type, params)))
    return tuple(normalized)


def _instance_input_ports(component_library: dict[str, dict[str, Any]], component_name: str) -> tuple[tuple[str, str], ...]:
    _, component = _component_definition(component_library, component_name)
    params = _component_params(component_library, component_name)
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
            inputs.append((str(port_name), resolve_component_type(port_type, params)))
    return tuple(inputs)


def _component_port_payload(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
    port_name: str,
) -> dict[str, Any] | None:
    _, component = _component_definition(component_library, component_name)
    ports = component.get("ports")
    if not isinstance(ports, dict):
        raise ValueError(f"component '{component_name}' must define object-valued 'ports'")
    payload = ports.get(port_name)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"component '{component_name}' port '{port_name}' must be an object")
    return payload


def _is_semantic_component_port_type(port_type: str) -> bool:
    return port_type in {"clock", "reset"}


def _semantic_component_port_signal(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
    port_name: str,
) -> str | None:
    payload = _component_port_payload(component_library, component_name, port_name)
    if payload is None:
        return None
    port_type = payload.get("type")
    if port_type == "clock":
        return "clk"
    if port_type == "reset":
        active = payload.get("active")
        return "rst" if active == "hi" else "rst_n"
    return None


def _needs_active_low_reset_helper(
    design: UHIRDesign,
    component_library: dict[str, dict[str, Any]],
) -> bool:
    for resource in design.resources:
        if resource.kind != "fu":
            continue
        for port_name, _port_type in _instance_ports(component_library, resource.value):
            payload = _component_port_payload(component_library, resource.value, port_name)
            if payload is None:
                continue
            if payload.get("type") == "reset" and payload.get("active") == "lo":
                return True
    return False


def _component_held_input_ports(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
) -> tuple[tuple[str, str], ...]:
    if not _component_should_hold_inputs(_component_kind(component_library, component_name)):
        return ()
    return _component_routed_input_ports(component_library, component_name)


def _component_routed_input_ports(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
) -> tuple[tuple[str, str], ...]:
    issue_bound_ports = set(_component_issue_bindings(component_library, component_name))
    return tuple(
        (port_name, port_type)
        for port_name, port_type in _instance_input_ports(component_library, component_name)
        if port_name not in issue_bound_ports
        and port_name not in _component_tied_input_ports(component_library, component_name)
        and not _is_semantic_component_port_type(port_type)
    )


def _component_tied_input_ports(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
) -> dict[str, str]:
    _, component = _component_definition(component_library, component_name)
    ports = component.get("ports")
    if not isinstance(ports, dict):
        raise ValueError(f"component '{component_name}' must define object-valued 'ports'")
    ties: dict[str, str] = {}
    for port_name, port_payload in ports.items():
        if not isinstance(port_payload, dict):
            raise ValueError(f"component '{component_name}' port '{port_name}' must be an object")
        if port_payload.get("dir") != "input":
            continue
        tie_expr = port_payload.get("tie")
        if isinstance(tie_expr, str) and tie_expr:
            ties[str(port_name)] = tie_expr
    return ties


def _component_should_hold_inputs(component_kind: str | None) -> bool:
    return component_kind in {"combinational", "memory"}


def _instance_output_ports(component_library: dict[str, dict[str, Any]], component_name: str) -> tuple[tuple[str, str], ...]:
    _, component = _component_definition(component_library, component_name)
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


def _add_semantic_value_result_nets(
    lowered: UGLIRDesign,
    design: UHIRDesign,
    component_library: dict[str, dict[str, Any]] | None,
) -> None:
    for value_id, producer_node in _producer_node_map(design).items():
        if not isinstance(value_id, str) or not value_id:
            continue
        result_type = getattr(producer_node, "result_type", None)
        if not isinstance(result_type, str) or not result_type:
            continue
        if not isinstance(getattr(producer_node, "attributes", {}).get("bind"), str):
            continue
        result_expr = _semantic_value_result_expr(design, value_id, producer_node, component_library)
        if result_expr is None or result_expr == value_id or _is_known_uglir_signal(lowered, value_id):
            continue
        lowered.resources.append(UGLIRResource("net", value_id, result_type))
        lowered.assigns.append(UGLIRAssign(value_id, result_expr))


def _semantic_value_result_expr(
    design: UHIRDesign,
    value_id: str,
    producer_node,
    component_library: dict[str, dict[str, Any]] | None,
) -> str | None:
    producer_region = _producer_region(design, value_id)
    direct_expr = _semantic_value_direct_result_expr(
        design,
        value_id,
        producer_node,
        component_library,
        region=producer_region,
    )
    if direct_expr is None:
        return None
    if design.stage != "fsm":
        return direct_expr

    top_controller = _require_top_level_controller(design)
    state_signal = _controller_state_id(top_controller, top_controller)
    timed_states = sorted(
        (
            int(state.name[1:]),
            state.attributes["code"],
        )
        for state in top_controller.states
        if isinstance(state.name, str)
        and re.fullmatch(r"T\d+", state.name)
        and isinstance(state.attributes.get("code"), int)
    )
    if not timed_states:
        return direct_expr

    timed_exprs: list[tuple[int, int, str]] = []
    for step, state_code in timed_states:
        step_expr = _resolve_value_signal(
            design,
            value_id,
            component_library,
            consumer_start=step,
            region=producer_region,
            prefer_semantic_value=False,
            allow_forward_handoff=False,
        )
        if step_expr is None or step_expr == value_id:
            step_expr = direct_expr
        timed_exprs.append((step, state_code, step_expr))

    default_expr = timed_exprs[-1][2]
    normalized_exprs = [(state_code, expr) for _step, state_code, expr in timed_exprs]

    unique_exprs = {expr for _state_code, expr in normalized_exprs}
    if len(unique_exprs) == 1:
        return normalized_exprs[0][1]

    branches = [
        f"{_state_eq_expr(state_signal, state_code)} ? {expr}"
        for state_code, expr in normalized_exprs
        if expr != default_expr
    ]
    if not branches:
        return default_expr
    return " : ".join(branches) + f" : {default_expr}"


def _semantic_value_direct_result_expr(
    design: UHIRDesign,
    value_id: str,
    producer_node,
    component_library: dict[str, dict[str, Any]] | None,
    *,
    region=None,
) -> str | None:
    identity_source = _identity_handoff_source(design, region, producer_node)
    if identity_source is not None:
        operand_region, operand_value = identity_source
        phi_carries = _phi_carry_specs(design)
        if operand_value in phi_carries:
            expr = _resolve_value_signal(
                design,
                operand_value,
                component_library,
                region=operand_region,
                prefer_semantic_value=False,
            )
        else:
            expr = _resolve_producer_signal(
                design,
                operand_value,
                component_library,
                region=operand_region,
                prefer_semantic_value=False,
            )
        if expr is None:
            expr = _resolve_value_signal(
                design,
                operand_value,
                component_library,
                region=operand_region,
                prefer_semantic_value=False,
            )
        if expr is not None:
            return expr
    return _node_result_signal(design, producer_node, component_library)


def _value_is_identity_handoff(design: UHIRDesign, value_id: str, *, region=None) -> bool:
    producer_region = region if region is not None and _region_producer_node(region, value_id) is not None else _producer_region(design, value_id)
    producer_node = _region_producer_node(producer_region, value_id)
    if producer_node is None:
        producer_node = _producer_node_map(design).get(value_id)
    if producer_node is None:
        return False
    return _identity_handoff_source(design, producer_region, producer_node) is not None


def _identity_handoff_source(design: UHIRDesign, producer_region, producer_node) -> tuple[object, str] | None:
    if producer_node.opcode == "mov" and len(producer_node.operands) == 1:
        operand_region, operand_value = _specialize_child_call_operand(design, producer_region, producer_node.operands[0])
        if isinstance(operand_value, str) and operand_value:
            return operand_region, operand_value
        return None
    if producer_node.opcode != "add" or len(producer_node.operands) != 2:
        return None
    result_type = getattr(producer_node, "result_type", None)
    if not isinstance(result_type, str) or not result_type:
        return None
    left, right = producer_node.operands
    if _is_zero_literal(left) and isinstance(right, str) and right:
        operand_region, operand_value = _specialize_child_call_operand(design, producer_region, right)
    elif _is_zero_literal(right) and isinstance(left, str) and left:
        operand_region, operand_value = _specialize_child_call_operand(design, producer_region, left)
    else:
        return None
    if _value_type(design, operand_value, region=operand_region) != result_type:
        return None
    return operand_region, operand_value


def _value_type(design: UHIRDesign, value_id: str, *, region=None) -> str | None:
    if not isinstance(value_id, str) or not value_id:
        return None
    literal_type = _literal_type(value_id)
    if literal_type is not None:
        return literal_type
    for port in (*design.inputs, *design.outputs):
        if port.name == value_id:
            return port.type
    for constant in design.constants:
        if constant.name == value_id:
            return constant.type
    producer_node = _region_producer_node(region, value_id)
    if producer_node is None:
        producer_node = _producer_node_map(design).get(value_id)
    result_type = getattr(producer_node, "result_type", None)
    if isinstance(result_type, str) and result_type:
        return result_type
    return None


def _literal_type(value_id: str) -> str | None:
    if value_id in {"true", "false"}:
        return "i1"
    if ":" not in value_id:
        return None
    _value_text, type_text = value_id.rsplit(":", 1)
    return type_text or None


def _is_zero_literal(operand: object) -> bool:
    if operand == 0:
        return True
    if not isinstance(operand, str):
        return False
    if operand == "0":
        return True
    return bool(re.fullmatch(r"0:[A-Za-z_][\w$<>]*", operand))


def _component_result_port(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
    opcode_name: str,
) -> str | None:
    _, component = _component_definition(component_library, component_name)
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


def _component_support_port_for_binding(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
    opcode_name: str,
    binding_value: str,
    *,
    require_direction: str | None = None,
) -> str | None:
    _, component = _component_definition(component_library, component_name)
    supports = component.get("supports")
    if not isinstance(supports, dict):
        raise ValueError(f"component '{component_name}' must define object-valued 'supports'")
    support = supports.get(opcode_name)
    if support is None:
        return None
    if not isinstance(support, dict):
        raise ValueError(f"component '{component_name}' support '{opcode_name}' must be an object")
    binding = support.get("bind")
    if binding is None:
        return None
    if not isinstance(binding, dict):
        raise ValueError(f"component '{component_name}' support '{opcode_name}' must use object-valued 'bind'")
    matched_ports = [port_name for port_name, binding_key in binding.items() if binding_key == binding_value]
    if require_direction is not None:
        matched_ports = [
            port_name
            for port_name in matched_ports
            if (_component_port_payload(component_library, component_name, port_name) or {}).get("dir") == require_direction
        ]
    if len(matched_ports) > 1:
        raise ValueError(
            f"component '{component_name}' support '{opcode_name}' must map at most one port to '{binding_value}'"
        )
    return matched_ports[0] if matched_ports else None


def _component_issue_bindings(
    component_library: dict[str, dict[str, Any]],
    component_name: str,
) -> dict[str, str]:
    _, component = _component_definition(component_library, component_name)
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
    _, component = _component_definition(component_library, component_name)
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
    node_region,
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
        operand_region, operand_value = _specialize_child_call_operand(design, node_region, node.operands[operand_index])
        consumer_start = node.attributes.get("start")
        operand_producer = _region_producer_node(operand_region, operand_value)
        if operand_producer is None:
            operand_producer = _producer_node_map(design).get(operand_value)
        consumer_bind = node.attributes.get("bind")
        if (
            isinstance(consumer_bind, str)
            and consumer_bind
            and operand_value in _phi_carry_specs(design)
            and isinstance(consumer_start, int)
        ):
            phi_node = operand_producer
            chosen = _phi_handoff_operand_for_step(design, phi_node, consumer_start) if phi_node is not None else None
            if isinstance(chosen, str) and chosen:
                chosen_region = _producer_region(design, chosen)
                chosen_producer = _region_producer_node(chosen_region, chosen)
                if chosen_producer is None:
                    chosen_producer = _producer_node_map(design).get(chosen)
                chosen_bind = None if chosen_producer is None else chosen_producer.attributes.get("bind")
                if isinstance(chosen_bind, str) and chosen_bind == consumer_bind:
                    if not _value_is_identity_handoff(design, chosen, region=chosen_region):
                        return _resolve_value_signal(
                            design,
                            operand_value,
                            component_library,
                            occurrence_index,
                            consumer_start=consumer_start,
                            region=operand_region,
                        )
        if operand_value in _phi_carry_specs(design) or (
            operand_producer is not None and operand_producer.opcode == "mov"
        ):
            handoff_expr = _resolve_handoff_source(
                design,
                operand_value,
                component_library,
                occurrence_index,
                consumer_start=consumer_start,
                region=operand_region,
            )
            if handoff_expr is not None:
                return handoff_expr
        return _resolve_value_signal(
            design,
            operand_value,
            component_library,
            occurrence_index,
            consumer_start=consumer_start,
            region=operand_region,
        )
    return binding_key


def _specialize_child_call_operand(design: UHIRDesign, node_region, value_id: str):
    if node_region is None or not isinstance(value_id, str) or not value_id:
        return node_region, value_id
    if _looks_like_literal(value_id) or value_id in {"true", "false"}:
        return node_region, value_id
    callers = [
        (region, node)
        for region in design.regions
        for node in region.nodes
        if node.opcode == "call" and node.attributes.get("child") == node_region.id
    ]
    if len(callers) != 1:
        return node_region, value_id
    formals = _region_external_operands(design, node_region)
    if value_id not in formals:
        return node_region, value_id
    caller_region, caller_node = callers[0]
    actuals = caller_node.operands[1:]
    formal_to_actual = {
        formal: actual
        for formal, actual in zip(formals, actuals, strict=False)
    }
    return caller_region, formal_to_actual.get(value_id, value_id)


def _region_external_operands(design: UHIRDesign, region) -> tuple[str, ...]:
    local_value_ids = {node.id for node in region.nodes}
    local_value_ids.update(mapping.source_id for mapping in region.mappings)
    global_ids = {port.name for port in [*design.inputs, *design.outputs]}
    global_ids.update(constant.name for constant in design.constants)

    externals: list[str] = []
    seen: set[str] = set()
    for node in region.nodes:
        operands = node.operands[1:] if node.opcode == "call" else node.operands
        for operand in operands:
            if not isinstance(operand, str) or not operand:
                continue
            if operand in seen or operand in local_value_ids or operand in global_ids:
                continue
            if operand in {"true", "false"} or _looks_like_literal(operand):
                continue
            seen.add(operand)
            externals.append(operand)
    return tuple(externals)


def _resolve_value_signal(
    design: UHIRDesign,
    value_id: str,
    component_library: dict[str, dict[str, Any]] | None,
    occurrence_index: int | None = None,
    consumer_start: int | None = None,
    region=None,
    *,
    prefer_semantic_value: bool = True,
    allow_forward_handoff: bool = True,
) -> str:
    local_producer = _region_producer_node(region, value_id)
    binding_value_id = value_id
    producer_region = region if local_producer is not None else _producer_region(design, value_id)
    register = _register_for_value_at_step(
        design,
        value_id,
        consumer_start,
        occurrence_index=occurrence_index,
        region=region,
    )
    if local_producer is not None and _register_for_value_at_step(
        design,
        local_producer.id,
        consumer_start,
        occurrence_index=occurrence_index,
        region=region,
    ) is not None:
        binding_value_id = local_producer.id
    phi_carries = _phi_carry_specs(design)
    if value_id in phi_carries:
        return phi_carries[value_id]["register"]
    if register is not None:
        live_starts = _value_live_starts(design, binding_value_id, region=region)
        capture_steps = {
            _value_capture_step(design, binding_value_id, live_start, component_library)
            for live_start in live_starts
        }
        if isinstance(consumer_start, int) and consumer_start in capture_steps:
            result_signal = _resolve_producer_signal(
                design,
                value_id,
                component_library,
                occurrence_index,
                consumer_start=consumer_start,
                region=region,
                prefer_semantic_value=prefer_semantic_value,
            )
            if result_signal is not None:
                return result_signal
        handoff_start = consumer_start
        if (
            allow_forward_handoff
            and
            component_library is not None
            and isinstance(consumer_start, int)
            and consumer_start not in live_starts
            and (consumer_start + 1) in live_starts
            and _value_may_bypass_register(design, binding_value_id, consumer_start + 1, component_library)
        ):
            handoff_start = consumer_start + 1
        if (
            isinstance(handoff_start, int)
            and handoff_start in live_starts
            and _value_may_bypass_register(design, binding_value_id, handoff_start, component_library)
        ):
            result_signal = _resolve_producer_signal(
                design,
                value_id,
                component_library,
                occurrence_index,
                consumer_start=handoff_start,
                region=region,
                prefer_semantic_value=prefer_semantic_value,
            )
            if result_signal is not None:
                return result_signal
        return register
    result_signal = _resolve_producer_signal(
        design,
        value_id,
        component_library,
        occurrence_index,
        consumer_start=consumer_start,
        region=region,
        prefer_semantic_value=prefer_semantic_value,
    )
    if result_signal is not None:
        return result_signal
    return value_id


def _resolve_handoff_source(
    design: UHIRDesign,
    value_id: str,
    component_library: dict[str, dict[str, Any]] | None,
    occurrence_index: int | None = None,
    consumer_start: int | None = None,
    region=None,
    *,
    _seen: set[tuple[str, int | None]] | None = None,
) -> str | None:
    if not isinstance(value_id, str) or not value_id:
        return None
    if not isinstance(consumer_start, int):
        return _resolve_value_signal(
            design,
            value_id,
            component_library,
            occurrence_index,
            consumer_start=consumer_start,
            region=region,
        )
    if _seen is None:
        _seen = set()
    key = (value_id, consumer_start)
    if key in _seen:
        return _resolve_value_signal(
            design,
            value_id,
            component_library,
            occurrence_index,
            consumer_start=consumer_start,
            region=region,
        )
    _seen.add(key)

    local_producer = _region_producer_node(region, value_id)
    producer_node = local_producer
    if producer_node is None:
        producer_node = _producer_node_map(design).get(value_id)
    if producer_node is None:
        return _resolve_value_signal(
            design,
            value_id,
            component_library,
            occurrence_index,
            consumer_start=consumer_start,
            region=region,
        )

    if value_id in _phi_carry_specs(design):
        if occurrence_index is None:
            chosen = _phi_handoff_operand_for_step(design, producer_node, consumer_start)
            if isinstance(chosen, str) and chosen:
                chosen_region = _producer_region(design, chosen)
                chosen_producer = _region_producer_node(chosen_region, chosen)
                if chosen_producer is None:
                    chosen_producer = _producer_node_map(design).get(chosen)
                chosen_live_steps = _value_global_live_starts(design, chosen)
                if chosen_region is not None:
                    chosen_live_steps = _value_live_starts(design, chosen, region=chosen_region)
                if (
                    consumer_start in chosen_live_steps
                    and not (
                        chosen_producer is not None
                        and chosen_producer.opcode in {"mov", "phi"}
                    )
                ):
                    chosen_expr = _resolve_producer_signal(
                        design,
                        chosen,
                        component_library,
                        occurrence_index,
                        consumer_start=consumer_start,
                        region=chosen_region,
                    )
                    if chosen_expr is not None:
                        return chosen_expr
                return _resolve_handoff_source(
                    design,
                    chosen,
                    component_library,
                    occurrence_index,
                    consumer_start=consumer_start,
                    region=chosen_region,
                    _seen=_seen,
                )
        return _resolve_value_signal(
            design,
            value_id,
            component_library,
            occurrence_index,
            consumer_start=consumer_start,
            region=region,
        )

    producer_region = region if local_producer is not None else _producer_region(design, value_id)
    identity_source = _identity_handoff_source(design, producer_region, producer_node)
    if identity_source is not None:
        operand_region, operand_value = identity_source
        return _resolve_handoff_source(
            design,
            operand_value,
            component_library,
            occurrence_index,
            consumer_start=consumer_start,
            region=operand_region,
            _seen=_seen,
        )

    producer_expr = _resolve_producer_signal(
        design,
        value_id,
        component_library,
        occurrence_index,
        consumer_start=consumer_start,
        region=region,
    )
    if producer_expr is not None:
        return producer_expr
    return _resolve_value_signal(
        design,
        value_id,
        component_library,
        occurrence_index,
        consumer_start=consumer_start,
        region=region,
    )


def _resolve_producer_signal(
    design: UHIRDesign,
    value_id: str,
    component_library: dict[str, dict[str, Any]] | None,
    occurrence_index: int | None = None,
    consumer_start: int | None = None,
    region=None,
    *,
    prefer_semantic_value: bool = True,
) -> str | None:
    local_producer = _region_producer_node(region, value_id)
    producer_node = local_producer
    if producer_node is None:
        producer_node = _producer_node_map(design).get(value_id)
    if producer_node is None:
        return None
    if (
        isinstance(value_id, str)
        and value_id
        and isinstance(getattr(producer_node, "result_type", None), str)
        and getattr(producer_node, "result_type")
        and isinstance(getattr(producer_node, "attributes", {}).get("bind"), str)
        and not (local_producer is not None and _value_id_is_ambiguous(design, value_id))
    ):
        if consumer_start is not None:
            producer_register = _register_for_value_at_step(
                design,
                value_id,
                consumer_start,
                occurrence_index=occurrence_index,
                region=region,
            )
            capture_steps = sorted(
                _producer_capture_steps_for_value(
                    design,
                    value_id,
                    component_library,
                    region=region if local_producer is not None else _producer_region(design, value_id),
                )
            )
            if (
                producer_register is not None
                and capture_steps
                and max(capture_steps) < consumer_start
            ):
                return producer_register
        if prefer_semantic_value:
            return value_id
        result_signal = _node_result_signal(design, producer_node, component_library)
        if result_signal is not None:
            return result_signal
    result_signal = _node_result_signal(design, producer_node, component_library)
    if result_signal is not None:
        return result_signal
    return _node_value_expr(
        design,
        producer_node,
        component_library,
        occurrence_index,
        consumer_start=consumer_start,
        region=region if local_producer is not None else _producer_region(design, value_id),
    )


def _resolve_phi_carry_update_source(
    design: UHIRDesign,
    value_id: str,
    step: int,
    component_library: dict[str, dict[str, Any]] | None,
    *,
    region=None,
) -> str | None:
    producer_node = _region_producer_node(region, value_id)
    if producer_node is None:
        producer_node = _producer_node_map(design).get(value_id)
    if producer_node is None and isinstance(value_id, str) and value_id:
        if value_id in {port.name for port in (*design.inputs, *design.outputs)}:
            return value_id
        if any(constant.name == value_id for constant in design.constants):
            return value_id
        if value_id in {"true", "false"} or _looks_like_literal(value_id):
            return value_id
        return None
    if producer_node is not None and producer_node.opcode == "phi":
        for operand in producer_node.operands:
            operand_region = _producer_region(design, operand)
            operand_steps: set[int]
            if operand_region is not None:
                operand_steps = _producer_global_capture_steps(
                    design,
                    operand_region.id,
                    operand,
                    component_library,
                )
            else:
                operand_steps = _value_global_live_starts(design, operand)
            if step not in operand_steps:
                continue
            source = _resolve_phi_carry_update_source(
                design,
                operand,
                step,
                component_library,
                region=operand_region,
            )
            if source is not None:
                return source
    producer_region = region if _region_producer_node(region, value_id) is not None else _producer_region(design, value_id)
    current_capture_steps: set[int] = set()
    if producer_region is not None:
        current_capture_steps = _producer_capture_steps_for_value(
            design,
            value_id,
            component_library,
            region=producer_region,
        )
    current_live_steps = _value_global_live_starts(design, value_id)
    if producer_node is not None:
        identity_source = None
        if step not in current_capture_steps and step not in current_live_steps:
            identity_source = _identity_handoff_source(design, producer_region, producer_node)
        if identity_source is not None:
            operand_region, operand_value = identity_source
            source = _resolve_phi_carry_update_source(
                design,
                operand_value,
                step,
                component_library,
                region=operand_region,
            )
            if source is not None:
                return source
    if (
        producer_node is not None
        and isinstance(value_id, str)
        and value_id
        and isinstance(getattr(producer_node, "result_type", None), str)
        and getattr(producer_node, "result_type")
        and isinstance(getattr(producer_node, "attributes", {}).get("bind"), str)
        and step in _value_global_live_starts(design, value_id)
        and not (region is not None and _value_id_is_ambiguous(design, value_id))
    ):
        return value_id
    if (
        producer_node is not None
        and isinstance(value_id, str)
        and value_id
        and isinstance(getattr(producer_node, "attributes", {}).get("bind"), str)
    ):
        producer_register = _register_for_value_at_step(design, value_id, step, region=region)
        local_live_starts = _value_live_starts(design, value_id, region=region) if region is not None else set()
        if (
            producer_register is not None
            and step not in local_live_starts
            and step in _producer_capture_steps_for_value(
                design,
                value_id,
                component_library,
                region=region,
            )
        ):
            return producer_register
    return _resolve_producer_signal(
        design,
        value_id,
        component_library,
        consumer_start=step,
        region=region,
    )


def _value_may_bypass_register(
    design: UHIRDesign,
    value_id: str,
    consumer_start: int,
    component_library: dict[str, dict[str, Any]] | None,
) -> bool:
    return _value_capture_step(design, value_id, consumer_start, component_library) == consumer_start


def _node_value_expr(
    design: UHIRDesign,
    node,
    component_library: dict[str, dict[str, Any]] | None,
    occurrence_index: int | None,
    consumer_start: int | None = None,
    region=None,
) -> str | None:
    def resolve_phi_choice(chosen: str) -> str:
        producer_expr = _resolve_producer_signal(
            design,
            chosen,
            component_library,
            None if occurrence_index is None else max(occurrence_index - 1, 0),
            consumer_start=consumer_start,
            region=_producer_region(design, chosen),
        )
        if producer_expr is not None:
            return producer_expr
        return _resolve_value_signal(
            design,
            chosen,
            component_library,
            None if occurrence_index is None else max(occurrence_index - 1, 0),
            consumer_start=consumer_start,
        )

    if node.opcode == "mov" and len(node.operands) == 1:
        operand_region, operand_value = _specialize_child_call_operand(design, region, node.operands[0])
        return _resolve_value_signal(
            design,
            operand_value,
            component_library,
            occurrence_index,
            consumer_start=consumer_start,
            region=operand_region,
        )
    if node.opcode == "phi":
        if not node.operands:
            return None
        branch_merge_expr = _branch_merge_phi_expr(
            design,
            node,
            component_library,
            occurrence_index,
            consumer_start=consumer_start,
            region=region,
        )
        if branch_merge_expr is not None:
            return branch_merge_expr
        if consumer_start is not None:
            incoming = tuple(node.attributes.get("incoming", ()))
            if (
                incoming
                and incoming[0] == "entry"
                and consumer_start == 0
                and (occurrence_index is None or occurrence_index == 0)
            ):
                chosen = node.operands[0]
                return resolve_phi_choice(chosen)
            best_index: int | None = None
            best_step: int | None = None
            for index, operand in enumerate(node.operands):
                operand_region = _producer_region(design, operand)
                operand_steps = sorted(_value_live_starts(design, operand, region=operand_region))
                for step in operand_steps:
                    if step > consumer_start:
                        continue
                    if best_step is None or step >= best_step:
                        best_index = index
                        best_step = step
            if best_index is not None:
                chosen = node.operands[best_index]
                return resolve_phi_choice(chosen)
        if occurrence_index is None or occurrence_index > 0:
            chosen = node.operands[1] if len(node.operands) > 1 else node.operands[0]
        else:
            chosen = node.operands[0]
        return resolve_phi_choice(chosen)
    if node.opcode == "call":
        returned_value = _call_return_value_id(design, node)
        if returned_value is None:
            return None
        child_region = design.get_region(node.attributes.get("child"))
        producer_signal = _resolve_producer_signal(
            design,
            returned_value,
            component_library,
            occurrence_index,
            consumer_start=consumer_start,
            region=child_region,
        )
        if producer_signal is not None:
            return producer_signal
        return _resolve_value_signal(
            design,
            returned_value,
            component_library,
            occurrence_index,
            consumer_start=consumer_start,
            region=child_region,
        )
    if node.opcode == "sel" and len(node.operands) == 3:
        condition = _resolve_value_signal(design, node.operands[0], component_library, occurrence_index)
        true_value = _resolve_value_signal(design, node.operands[1], component_library, occurrence_index)
        false_value = _resolve_value_signal(design, node.operands[2], component_library, occurrence_index)
        return f"{condition} ? {true_value} : {false_value}"
    return None


def _phi_handoff_operand_for_step(
    design: UHIRDesign,
    node,
    step: int,
) -> str | None:
    if node.opcode != "phi" or not node.operands:
        return None
    incoming = tuple(node.attributes.get("incoming", ()))
    if incoming and incoming[0] == "entry" and step == 0:
        return node.operands[0]

    best_index: int | None = None
    best_step: int | None = None
    for index, operand in enumerate(node.operands):
        operand_steps = sorted(_value_global_live_starts(design, operand))
        for operand_step in operand_steps:
            if operand_step > step:
                continue
            if best_step is None or operand_step >= best_step:
                best_index = index
                best_step = operand_step
    if best_index is not None:
        return node.operands[best_index]
    return node.operands[1] if len(node.operands) > 1 else node.operands[0]


def _producer_capture_steps_for_value(
    design: UHIRDesign,
    value_id: str,
    component_library: dict[str, dict[str, Any]] | None,
    *,
    region=None,
) -> set[int]:
    if not isinstance(value_id, str) or not value_id:
        return set()
    producer_region = region if region is not None else _producer_region(design, value_id)
    if producer_region is None:
        return _value_global_live_starts(design, value_id)
    return _producer_global_capture_steps(
        design,
        producer_region.id,
        value_id,
        component_library,
    )


def _call_return_value_id(design: UHIRDesign, node) -> str | None:
    child_id = node.attributes.get("child")
    if not isinstance(child_id, str) or not child_id:
        return None
    child_region = next((region for region in design.regions if region.id == child_id), None)
    if child_region is None:
        return None
    returned_values = [
        child_node.operands[0]
        for child_node in child_region.nodes
        if child_node.opcode == "ret" and child_node.operands
    ]
    if len(returned_values) != 1:
        return None
    return returned_values[0]


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


def _region_producer_node(region, value_id: str):
    if region is None or not isinstance(value_id, str) or not value_id:
        return None
    local_nodes = {node.id: node for node in region.nodes}
    node = local_nodes.get(value_id)
    if node is not None:
        return node
    for source_map in region.mappings:
        if source_map.source_id != value_id:
            continue
        node = local_nodes.get(source_map.node_id)
        if node is not None:
            return node
    return None


def _producer_region(design: UHIRDesign, value_id: str):
    if not isinstance(value_id, str) or not value_id:
        return None
    for region in design.regions:
        if _region_producer_node(region, value_id) is not None:
            return region
    return None


def _value_id_is_ambiguous(design: UHIRDesign, value_id: str) -> bool:
    if not isinstance(value_id, str) or not value_id:
        return False
    matches = 0
    for region in design.regions:
        local_nodes = {node.id for node in region.nodes}
        if value_id in local_nodes:
            matches += 1
        for source_map in region.mappings:
            if source_map.source_id == value_id and source_map.node_id in local_nodes:
                matches += 1
        if matches > 1:
            return True
    return False


def _default_net_expr(port_type: str) -> str:
    if port_type == "i1":
        return "false"
    return f"0:{port_type}"


def _branch_merge_phi_expr(
    design: UHIRDesign,
    node,
    component_library: dict[str, dict[str, Any]] | None,
    occurrence_index: int | None,
    *,
    consumer_start: int | None = None,
    region=None,
) -> str | None:
    incoming = tuple(node.attributes.get("incoming", ()))
    if len(node.operands) != 2 or len(incoming) != 2:
        return None
    node_region = region if region is not None else _producer_region(design, node.id)
    if node_region is None:
        return None

    phi_start = node.attributes.get("start")
    best_branch = None
    best_start = -1
    for candidate in node_region.nodes:
        if candidate.opcode != "branch" or not candidate.operands:
            continue
        true_label = candidate.attributes.get("true_input_label")
        false_label = candidate.attributes.get("false_input_label")
        if not isinstance(true_label, str) or not isinstance(false_label, str):
            continue
        if {true_label, false_label} != set(incoming):
            continue
        candidate_start = candidate.attributes.get("start")
        if isinstance(phi_start, int) and isinstance(candidate_start, int) and candidate_start > phi_start:
            continue
        if isinstance(candidate_start, int) and candidate_start >= best_start:
            best_branch = candidate
            best_start = candidate_start
    if best_branch is None:
        return None

    true_label = best_branch.attributes.get("true_input_label")
    false_label = best_branch.attributes.get("false_input_label")
    assert isinstance(true_label, str)
    assert isinstance(false_label, str)

    try:
        true_index = incoming.index(true_label)
        false_index = incoming.index(false_label)
    except ValueError:
        return None

    branch_start = best_branch.attributes.get("start")
    branch_consumer_start = consumer_start if isinstance(consumer_start, int) else branch_start if isinstance(branch_start, int) else None
    condition_expr = _resolve_value_signal(
        design,
        best_branch.operands[0],
        component_library,
        occurrence_index,
        consumer_start=branch_consumer_start,
        region=node_region,
    )
    true_expr = _resolve_value_signal(
        design,
        node.operands[true_index],
        component_library,
        occurrence_index,
        consumer_start=consumer_start,
        region=node_region,
    )
    false_expr = _resolve_value_signal(
        design,
        node.operands[false_index],
        component_library,
        occurrence_index,
        consumer_start=consumer_start,
        region=node_region,
    )
    return f"({condition_expr}) ? {true_expr} : {false_expr}"


def _opcode_expr(
    design: UHIRDesign,
    controllers,
    top_controller,
    resource_id: str,
    component_name: str,
    controller_codes: dict[str, dict[str, int]],
    component_library: dict[str, dict[str, Any]],
) -> str:
    _, component = _component_definition(component_library, component_name)
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
