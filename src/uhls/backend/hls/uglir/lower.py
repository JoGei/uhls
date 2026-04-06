"""FSM-to-uglir lowering entrypoint."""

from __future__ import annotations

from math import ceil, log2

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


def lower_fsm_to_uglir(design: UHIRDesign) -> UHIRDesign:
    """Lower one static fsm-stage µhIR design to one initial uglir shell."""
    if design.stage != "fsm":
        raise ValueError(f"uglir lowering expects fsm-stage µhIR input, got stage '{design.stage}'")
    if any(controller.attributes.get("protocol") != "req_resp" for controller in design.controllers):
        raise ValueError("initial uglir lowering currently supports only one static top-level req_resp controller")
    if len(design.controllers) != 1:
        raise ValueError("initial uglir lowering currently expects exactly one controller")

    controller = design.controllers[0]
    lowered = UHIRDesign(name=design.name, stage="uglir")
    lowered.inputs = [
        UHIRPort("input", "clk", "clock"),
        UHIRPort("input", "rst", "i1"),
        UHIRPort("input", "req_valid", "i1"),
        UHIRPort("input", "resp_ready", "i1"),
    ]
    lowered.inputs.extend(UHIRPort(port.direction, port.name, port.type) for port in design.inputs)
    lowered.outputs = [
        UHIRPort("output", "req_ready", "i1"),
        UHIRPort("output", "resp_valid", "i1"),
    ]
    lowered.outputs.extend(UHIRPort(port.direction, port.name, port.type) for port in design.outputs)
    lowered.constants = list(design.constants)

    state_type = _state_type(controller)
    lowered.resources.append(UHIRResource("reg", "state", state_type))
    lowered.resources.append(UHIRResource("net", "next_state", state_type))
    lowered.resources.append(UHIRResource("net", "req_fire", "i1"))
    lowered.resources.append(UHIRResource("net", "resp_fire", "i1"))

    for resource in design.resources:
        if resource.kind == "fu":
            lowered.resources.append(UHIRResource("inst", resource.id, resource.value))
            lowered.resources.append(UHIRResource("net", f"{resource.id}_go", "i1"))
            result_type = _instance_result_type(design, resource.id)
            if result_type is not None:
                lowered.resources.append(UHIRResource("net", f"{resource.id}_y", result_type))
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
        if resource.kind == "fu":
            lowered.assigns.append(UHIRAssign(f"{resource.id}_go", _issue_expr(controller, resource.id, state_code)))
            lowered.attachments.append(UHIRAttach(resource.id, "go", f"{resource.id}_go"))
            if any(candidate.id == f"{resource.id}_y" for candidate in lowered.resources):
                lowered.attachments.append(UHIRAttach(resource.id, "y", f"{resource.id}_y"))

    for register in latch_targets:
        lowered.assigns.append(UHIRAssign(f"latch_{register}", _latch_expr(controller, register, state_code)))
        lowered.assigns.append(UHIRAssign(f"sel_{register}", _select_expr(design, controller, register, state_code)))
        lowered.glue_muxes.append(_build_register_mux(design, register))

    for output_name, driver in _output_drivers(design).items():
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
        for issue in attrs.get("issue", ())
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


def _select_expr(design: UHIRDesign, controller, register: str, state_code: dict[str, int]) -> str:
    choices = _register_state_sources(design, controller, register)
    if not choices:
        return "hold"
    branches = [f"{_state_eq_expr(state_code[state])} ? {source}" for state, source in sorted(choices.items(), key=lambda item: state_code[item[0]])]
    return " : ".join(branches) + " : hold"


def _build_register_mux(design: UHIRDesign, register: str) -> UHIRGlueMux:
    register_type = _resource_value(design, register)
    glue_mux = UHIRGlueMux(name=f"mx_{register}", type=register_type, select=f"sel_{register}")
    glue_mux.cases.append(UHIRGlueMuxCase("hold", register))
    seen_sources = {register}
    for source in _register_possible_sources(design, register):
        if source in seen_sources:
            continue
        glue_mux.cases.append(UHIRGlueMuxCase(source, source))
        seen_sources.add(source)
    return glue_mux


def _register_possible_sources(design: UHIRDesign, register: str) -> list[str]:
    producer_to_instance = _producer_instance_map(design)
    sources: list[str] = []
    for region in design.regions:
        for binding in region.value_bindings:
            if binding.register != register:
                continue
            instance = producer_to_instance.get(binding.producer)
            if instance is not None:
                sources.append(f"{instance}_y")
    return sources


def _register_state_sources(design: UHIRDesign, controller, register: str) -> dict[str, str]:
    producer_to_instance = _producer_instance_map(design)
    live_start_to_instance: dict[int, str] = {}
    for region in design.regions:
        for binding in region.value_bindings:
            if binding.register != register:
                continue
            instance = producer_to_instance.get(binding.producer)
            if instance is None:
                continue
            live_start_to_instance[binding.live_start] = f"{instance}_y"
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


def _output_drivers(design: UHIRDesign) -> dict[str, str]:
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
            drivers[output.name] = f"{producer_to_instance[returned]}_y"
        else:
            drivers[output.name] = returned
    return drivers
