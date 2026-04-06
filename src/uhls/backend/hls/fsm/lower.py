"""Bind-to-FSM lowering entrypoint."""

from __future__ import annotations

from collections import defaultdict

from uhls.utils.graph import topological_sort
from uhls.backend.uhir.timing import TimingExpr
from uhls.backend.uhir.model import (
    UHIRConstant,
    UHIRController,
    UHIRControllerEmit,
    UHIRControllerLink,
    UHIRControllerState,
    UHIRControllerTransition,
    UHIRDesign,
    UHIREdge,
    UHIRMux,
    UHIRNode,
    UHIRPort,
    UHIRRegion,
    UHIRRegionRef,
    UHIRResource,
    UHIRSchedule,
    UHIRSourceMap,
    UHIRValueBinding,
)

FSM_ENCODINGS = ("binary", "one_hot")


def lower_bind_to_fsm(design: UHIRDesign, *, encoding: str = "binary") -> UHIRDesign:
    """Lower one bind-stage µhIR design to an fsm-stage controller shell."""
    if design.stage != "bind":
        raise ValueError(f"fsm lowering expects bind-stage µhIR input, got stage '{design.stage}'")
    normalized_encoding = encoding.strip().lower().replace("-", "_")
    if normalized_encoding not in FSM_ENCODINGS:
        supported = ", ".join(FSM_ENCODINGS)
        raise ValueError(f"unsupported fsm encoding '{encoding}'; expected one of: {supported}")
    symbolic = _design_has_symbolic_timing(design)
    if symbolic:
        _validate_supported_dynamic_bind_timing(design)
    else:
        _validate_concrete_bind_timing(design)

    lowered = UHIRDesign(name=design.name, stage="fsm")
    lowered.inputs = [_clone_port(port) for port in design.inputs]
    lowered.outputs = [_clone_port(port) for port in design.outputs]
    lowered.constants = [_clone_constant(constant) for constant in design.constants]
    lowered.schedule = None if design.schedule is None else UHIRSchedule(design.schedule.kind)
    lowered.resources = [UHIRResource(resource.kind, resource.id, resource.value, resource.target) for resource in design.resources]
    if symbolic:
        lowered.controllers = [_build_dynamic_controller(design, normalized_encoding)]
        lowered.controllers.extend(_build_dynamic_child_controllers(design, normalized_encoding))
    else:
        lowered.controllers = [_build_static_controller(design, normalized_encoding)]
    lowered.regions = [_clone_fsm_region(region) for region in design.regions]
    return lowered


def _validate_concrete_bind_timing(design: UHIRDesign) -> None:
    """Reject symbolic bind timing until dynamic FSM synthesis exists."""
    for region in design.regions:
        if region.steps is not None and any(isinstance(value, TimingExpr) for value in region.steps):
            raise ValueError(
                f"fsm lowering currently requires concrete bind timing; region '{region.id}' has symbolic steps"
            )
        if isinstance(region.latency, TimingExpr):
            raise ValueError(
                f"fsm lowering currently requires concrete bind timing; region '{region.id}' has symbolic latency"
            )
        if isinstance(region.initiation_interval, TimingExpr):
            raise ValueError(
                f"fsm lowering currently requires concrete bind timing; region '{region.id}' has symbolic ii"
            )
        for node in region.nodes:
            delay = node.attributes.get("delay")
            start = node.attributes.get("start")
            end = node.attributes.get("end")
            if isinstance(delay, TimingExpr) or isinstance(start, TimingExpr) or isinstance(end, TimingExpr):
                raise ValueError(
                    "fsm lowering currently requires concrete bind timing; "
                    f"node '{region.id}/{node.id}' still carries symbolic delay/start/end"
                )


def _validate_supported_dynamic_bind_timing(design: UHIRDesign) -> None:
    """Reject dynamic bind features that the current recursive FSM model cannot consume."""
    root_regions = [region for region in design.regions if region.parent is None]
    if len(root_regions) != 1:
        raise ValueError("dynamic fsm lowering currently expects exactly one top-level scheduled region")


def _build_dynamic_controller(design: UHIRDesign, encoding: str) -> UHIRController:
    """Build one coarse-grained dynamic controller over one symbolic top-level region."""
    # TODO: recursive dynamic FSM synthesis should introduce child controllers with
    # the internal handshake {act_valid, done_ready} -> {act_ready, done_valid} instead of
    # flattening all dynamic hierarchy progress into this one coarse controller.
    root_region = next(region for region in design.regions if region.parent is None)
    phases = _collect_dynamic_phases(root_region)
    phase_ids = sorted(phases)

    state_names = ["IDLE"]
    for phase_id in phase_ids:
        state_names.append(f"P{phase_id}")
        symbolic_nodes = [node for node in phases[phase_id] if _is_symbolic_hierarchy_node(node)]
        if symbolic_nodes:
            if len(symbolic_nodes) == 1:
                state_names.append(f"WAIT_{symbolic_nodes[0].id}")
            else:
                state_names.append(f"WAIT_P{phase_id}")
    state_names.append("DONE")

    states = [
        UHIRControllerState(name=state_name, attributes={"code": _state_code(index, encoding)})
        for index, state_name in enumerate(state_names)
    ]

    transitions: list[UHIRControllerTransition] = []
    emits: list[UHIRControllerEmit] = [
        UHIRControllerEmit("IDLE", {"req_ready": True}),
        UHIRControllerEmit("DONE", {"resp_valid": True}),
    ]

    first_phase = phase_ids[0] if phase_ids else None
    if first_phase is None:
        transitions.append(UHIRControllerTransition("IDLE", "DONE", {"when": "req_valid && req_ready"}))
    else:
        first_when = _conjoin_conditions(("req_valid && req_ready", _dynamic_phase_ready_condition(phases[first_phase])))
        transitions.append(UHIRControllerTransition("IDLE", f"P{first_phase}", {"when": first_when}))

    for index, phase_id in enumerate(phase_ids):
        phase_state = f"P{phase_id}"
        phase_nodes = phases[phase_id]
        symbolic_nodes = [node for node in phase_nodes if _is_symbolic_hierarchy_node(node)]
        issue_actions = sorted(
            f"{bind}<-{node.id}"
            for node in phase_nodes
            if isinstance((bind := node.attributes.get("bind")), str)
            and isinstance(node.attributes.get("class"), str)
            and node.attributes.get("class") != "CTRL"
        )
        activate_actions = sorted(node.id for node in symbolic_nodes)
        emit_attrs: dict[str, object] = {}
        if issue_actions:
            emit_attrs["issue"] = tuple(issue_actions)
        if activate_actions:
            emit_attrs["activate"] = tuple(activate_actions)
        if emit_attrs:
            emits.append(UHIRControllerEmit(phase_state, emit_attrs))

        next_state = "DONE" if index == len(phase_ids) - 1 else f"P{phase_ids[index + 1]}"
        if symbolic_nodes:
            wait_state = f"WAIT_{symbolic_nodes[0].id}" if len(symbolic_nodes) == 1 else f"WAIT_P{phase_id}"
            transitions.append(UHIRControllerTransition(phase_state, wait_state))
            wait_when = _dynamic_phase_completion_condition(symbolic_nodes)
            transition_attrs = {} if wait_when is None else {"when": wait_when}
            transitions.append(UHIRControllerTransition(wait_state, next_state, transition_attrs))
        else:
            ready_when = _dynamic_phase_ready_condition(phases[phase_ids[index + 1]]) if next_state.startswith("P") else None
            transition_attrs = {} if ready_when is None else {"when": ready_when}
            transitions.append(UHIRControllerTransition(phase_state, next_state, transition_attrs))

    transitions.append(UHIRControllerTransition("DONE", "IDLE", {"when": "resp_valid && resp_ready"}))

    return UHIRController(
        name="C0",
        attributes={
            "encoding": encoding,
            "protocol": "req_resp",
            "completion_order": "in_order",
            "overlap": True,
            "region": root_region.id,
        },
        inputs=[
            UHIRPort("input", "req_valid", "i1"),
            UHIRPort("input", "resp_ready", "i1"),
        ],
        outputs=[
            UHIRPort("output", "req_ready", "i1"),
            UHIRPort("output", "resp_valid", "i1"),
        ],
        states=states,
        transitions=transitions,
        emits=emits,
        links=_build_dynamic_top_level_links(root_region),
    )


def _build_dynamic_child_controllers(design: UHIRDesign, encoding: str) -> list[UHIRController]:
    """Build recursive child controllers for supported dynamic hierarchy nodes."""
    region_by_id = {region.id: region for region in design.regions}
    controllers: list[UHIRController] = []
    seen_regions: set[str] = set()
    for region in design.regions:
        for node in region.nodes:
            if node.attributes.get("timing") != "symbolic":
                continue
            child_id = node.attributes.get("child")
            if node.opcode == "loop":
                if not isinstance(child_id, str) or child_id in seen_regions:
                    continue
                child_region = region_by_id.get(child_id)
                if child_region is None:
                    continue
                controllers.append(_build_dynamic_loop_child_controller(node, child_region, region_by_id, encoding))
                seen_regions.add(child_id)
            elif node.opcode == "call":
                if not isinstance(child_id, str) or child_id in seen_regions:
                    continue
                child_region = region_by_id.get(child_id)
                if child_region is None:
                    continue
                controllers.append(_build_dynamic_call_child_controller(node, child_region, region_by_id, encoding))
                seen_regions.add(child_id)
            elif node.opcode == "branch":
                if node.id in seen_regions:
                    continue
                controllers.append(_build_dynamic_branch_child_controller(node, region_by_id, encoding))
                seen_regions.add(node.id)
            else:
                continue
    return controllers


def _build_dynamic_top_level_links(root_region: UHIRRegion) -> list[UHIRControllerLink]:
    """Build explicit parent-to-child controller links for one dynamic top-level controller."""
    links: list[UHIRControllerLink] = []
    for node in root_region.nodes:
        if node.opcode not in {"loop", "call", "branch"} or node.attributes.get("timing") != "symbolic":
            continue
        if node.opcode == "branch":
            child_name = f"C_{node.id}"
        else:
            child_id = node.attributes.get("child")
            if not isinstance(child_id, str):
                continue
            child_name = f"C_{child_id}"
        links.append(
            UHIRControllerLink(
                child=child_name,
                node=node.id,
                attributes={
                    "act": ("activate", "act_valid"),
                    "ready": ("ready", "act_ready"),
                    "done": ("completion", "done_valid"),
                    "done_ready": ("resp_ready", "done_ready"),
                },
            )
        )
    return links


def _build_dynamic_branch_child_controller(
    branch_node: UHIRNode,
    region_by_id: dict[str, UHIRRegion],
    encoding: str,
) -> UHIRController:
    """Build one recursive controller for one dynamic branch node."""
    true_child = branch_node.attributes.get("true_child")
    false_child = branch_node.attributes.get("false_child")
    if not isinstance(true_child, str) or not isinstance(false_child, str):
        raise ValueError(f"dynamic branch node '{branch_node.id}' must declare true_child/false_child")
    condition = branch_node.attributes.get("branch_condition")
    if not isinstance(condition, str) or not condition:
        raise ValueError(f"dynamic branch node '{branch_node.id}' is missing branch_condition=...")

    true_max_step = _concrete_subtree_max_end(true_child, region_by_id)
    false_max_step = _concrete_subtree_max_end(false_child, region_by_id)
    state_names = [
        "IDLE",
        *[f"TRUE_T{time_step}" for time_step in range(true_max_step + 1)],
        *[f"FALSE_T{time_step}" for time_step in range(false_max_step + 1)],
        "DONE",
    ]
    states = [
        UHIRControllerState(name=state_name, attributes={"code": _state_code(index, encoding)})
        for index, state_name in enumerate(state_names)
    ]

    transitions: list[UHIRControllerTransition] = [
        UHIRControllerTransition("IDLE", "TRUE_T0", {"when": f"act_valid && act_ready && {condition}"}),
        UHIRControllerTransition("IDLE", "FALSE_T0", {"when": f"act_valid && act_ready && !{condition}"}),
    ]
    for time_step in range(true_max_step):
        transitions.append(UHIRControllerTransition(f"TRUE_T{time_step}", f"TRUE_T{time_step + 1}"))
    transitions.append(UHIRControllerTransition(f"TRUE_T{true_max_step}", "DONE"))
    for time_step in range(false_max_step):
        transitions.append(UHIRControllerTransition(f"FALSE_T{time_step}", f"FALSE_T{time_step + 1}"))
    transitions.append(UHIRControllerTransition(f"FALSE_T{false_max_step}", "DONE"))
    transitions.append(UHIRControllerTransition("DONE", "IDLE", {"when": "done_valid && done_ready"}))

    true_actions = _collect_region_subtree_actions(true_child, region_by_id)
    false_actions = _collect_region_subtree_actions(false_child, region_by_id)
    emits: list[UHIRControllerEmit] = [
        UHIRControllerEmit("IDLE", {"act_ready": True}),
        UHIRControllerEmit("DONE", {"done_valid": True}),
    ]
    for time_step in range(true_max_step + 1):
        attrs = _emit_attrs_for_actions(true_actions.get(time_step, {}))
        if attrs:
            emits.append(UHIRControllerEmit(f"TRUE_T{time_step}", attrs))
    for time_step in range(false_max_step + 1):
        attrs = _emit_attrs_for_actions(false_actions.get(time_step, {}))
        if attrs:
            emits.append(UHIRControllerEmit(f"FALSE_T{time_step}", attrs))

    return UHIRController(
        name=f"C_{branch_node.id}",
        attributes={
            "encoding": encoding,
            "protocol": "act_done",
            "completion_order": "in_order",
            "overlap": True,
            "region": true_child,
            "false_region": false_child,
            "parent_node": branch_node.id,
            "branch_condition": condition,
        },
        inputs=[
            UHIRPort("input", "act_valid", "i1"),
            UHIRPort("input", "done_ready", "i1"),
        ],
        outputs=[
            UHIRPort("output", "act_ready", "i1"),
            UHIRPort("output", "done_valid", "i1"),
        ],
        states=states,
        transitions=transitions,
        emits=emits,
    )


def _build_dynamic_call_child_controller(
    call_node: UHIRNode,
    child_region: UHIRRegion,
    region_by_id: dict[str, UHIRRegion],
    encoding: str,
) -> UHIRController:
    """Build one recursive controller for one dynamic call child region."""
    if child_region.steps is not None and all(isinstance(value, int) for value in child_region.steps):
        _, max_step = child_region.steps
    else:
        max_step = _concrete_subtree_max_end(child_region.id, region_by_id)
    state_names = ["IDLE", *[f"T{time_step}" for time_step in range(max_step + 1)], "DONE"]
    states = [
        UHIRControllerState(name=state_name, attributes={"code": _state_code(index, encoding)})
        for index, state_name in enumerate(state_names)
    ]

    transitions: list[UHIRControllerTransition] = [
        UHIRControllerTransition("IDLE", "T0", {"when": "act_valid && act_ready"})
    ]
    for time_step in range(max_step):
        transitions.append(UHIRControllerTransition(f"T{time_step}", f"T{time_step + 1}"))
    transitions.append(UHIRControllerTransition(f"T{max_step}", "DONE"))
    transitions.append(UHIRControllerTransition("DONE", "IDLE", {"when": "done_valid && done_ready"}))

    subtree_actions = _collect_region_subtree_actions(child_region.id, region_by_id)
    emits: list[UHIRControllerEmit] = [
        UHIRControllerEmit("IDLE", {"act_ready": True}),
        UHIRControllerEmit("DONE", {"done_valid": True}),
    ]
    for time_step in range(max_step + 1):
        issues = sorted(subtree_actions.get(time_step, {}).get("issue", ()))
        latches = sorted(subtree_actions.get(time_step, {}).get("latch", ()))
        selects = sorted(subtree_actions.get(time_step, {}).get("select", ()))
        attrs = _emit_attrs_from_lists(issues, latches, selects)
        if attrs:
            emits.append(UHIRControllerEmit(f"T{time_step}", attrs))

    return UHIRController(
        name=f"C_{child_region.id}",
        attributes={
            "encoding": encoding,
            "protocol": "act_done",
            "completion_order": "in_order",
            "overlap": True,
            "region": child_region.id,
            "parent_node": call_node.id,
        },
        inputs=[
            UHIRPort("input", "act_valid", "i1"),
            UHIRPort("input", "done_ready", "i1"),
        ],
        outputs=[
            UHIRPort("output", "act_ready", "i1"),
            UHIRPort("output", "done_valid", "i1"),
        ],
        states=states,
        transitions=transitions,
        emits=emits,
    )


def _build_static_controller(design: UHIRDesign, encoding: str) -> UHIRController:
    """Build the first concrete static control-step controller."""
    max_time = _design_max_end(design)
    state_names = ["IDLE", *[f"T{time_step}" for time_step in range(max_time + 1)], "DONE"]
    states = [
        UHIRControllerState(name=state_name, attributes={"code": _state_code(index, encoding)})
        for index, state_name in enumerate(state_names)
    ]

    transitions: list[UHIRControllerTransition] = [
        UHIRControllerTransition("IDLE", "T0", {"when": "req_valid && req_ready"})
    ]
    for time_step in range(max_time):
        transitions.append(UHIRControllerTransition(f"T{time_step}", f"T{time_step + 1}"))
    transitions.append(UHIRControllerTransition(f"T{max_time}", "DONE"))
    transitions.append(UHIRControllerTransition("DONE", "IDLE", {"when": "resp_valid && resp_ready"}))

    emits: list[UHIRControllerEmit] = [
        UHIRControllerEmit("IDLE", {"req_ready": True}),
        UHIRControllerEmit("DONE", {"resp_valid": True}),
    ]
    for time_step, actions in sorted(_collect_time_step_actions(design).items()):
        if not actions:
            continue
        attrs: dict[str, object] = {}
        if actions["issue"]:
            attrs["issue"] = tuple(actions["issue"])
        if actions["latch"]:
            attrs["latch"] = tuple(actions["latch"])
        if actions["select"]:
            attrs["select"] = tuple(actions["select"])
        emits.append(UHIRControllerEmit(f"T{time_step}", attrs))

    return UHIRController(
        name="C0",
        attributes={
            "encoding": encoding,
            "protocol": "req_resp",
            "completion_order": "in_order",
            "overlap": True,
            "region": next((region.id for region in design.regions if region.parent is None), ""),
        },
        inputs=[
            UHIRPort("input", "req_valid", "i1"),
            UHIRPort("input", "resp_ready", "i1"),
        ],
        outputs=[
            UHIRPort("output", "req_ready", "i1"),
            UHIRPort("output", "resp_valid", "i1"),
        ],
        states=states,
        transitions=transitions,
        emits=emits,
    )


def _build_dynamic_loop_child_controller(
    loop_node: UHIRNode,
    header_region: UHIRRegion,
    region_by_id: dict[str, UHIRRegion],
    encoding: str,
) -> UHIRController:
    """Build one recursive controller for one dynamic loop-header region."""
    if header_region.steps is None or not all(isinstance(value, int) for value in header_region.steps):
        raise ValueError(
            f"dynamic loop child controller currently requires concrete local header timing; region '{header_region.id}' is missing integer steps"
        )
    branch = next((node for node in header_region.nodes if node.opcode == "branch"), None)
    if branch is None:
        raise ValueError(f"dynamic loop child controller requires a branch in loop region '{header_region.id}'")
    branch_start = branch.attributes.get("start")
    if not isinstance(branch_start, int):
        raise ValueError(f"loop header branch '{header_region.id}/{branch.id}' requires one concrete start time")
    true_child = branch.attributes.get("true_child")
    if not isinstance(true_child, str):
        raise ValueError(f"loop header branch '{header_region.id}/{branch.id}' must declare true_child")

    _, max_step = header_region.steps
    state_names = ["IDLE", *[f"T{time_step}" for time_step in range(max_step + 1)], "DONE"]
    states = [
        UHIRControllerState(name=state_name, attributes={"code": _state_code(index, encoding)})
        for index, state_name in enumerate(state_names)
    ]

    transitions: list[UHIRControllerTransition] = [
        UHIRControllerTransition("IDLE", "T0", {"when": "act_valid && act_ready"})
    ]
    for time_step in range(max(branch_start - 1, 0)):
        transitions.append(UHIRControllerTransition(f"T{time_step}", f"T{time_step + 1}"))
    if branch_start == 0:
        transitions.append(UHIRControllerTransition("T0", "DONE", {"when": _require_loop_condition(loop_node, "exit_when")}))
        if max_step >= 0:
            transitions.append(UHIRControllerTransition("T0", "T0", {"when": _require_loop_condition(loop_node, "iterate_when")}))
    else:
        decision_source = f"T{branch_start - 1}"
        decision_target = f"T{branch_start}"
        transitions.append(UHIRControllerTransition(decision_source, decision_target, {"when": _require_loop_condition(loop_node, "iterate_when")}))
        transitions.append(UHIRControllerTransition(decision_source, "DONE", {"when": _require_loop_condition(loop_node, "exit_when")}))
        for time_step in range(branch_start, max_step):
            transitions.append(UHIRControllerTransition(f"T{time_step}", f"T{time_step + 1}"))
        transitions.append(UHIRControllerTransition(f"T{max_step}", "T0"))
    transitions.append(UHIRControllerTransition("DONE", "IDLE", {"when": "done_valid && done_ready"}))

    header_actions = _collect_region_local_actions(header_region)
    body_actions = _collect_region_subtree_actions(true_child, region_by_id)
    emits: list[UHIRControllerEmit] = [
        UHIRControllerEmit("IDLE", {"act_ready": True}),
        UHIRControllerEmit("DONE", {"done_valid": True}),
    ]
    for time_step in range(max_step + 1):
        issues = sorted([*header_actions.get(time_step, {}).get("issue", ()), *body_actions.get(time_step, {}).get("issue", ())])
        latches = sorted([*header_actions.get(time_step, {}).get("latch", ()), *body_actions.get(time_step, {}).get("latch", ())])
        selects = sorted([*header_actions.get(time_step, {}).get("select", ()), *body_actions.get(time_step, {}).get("select", ())])
        attrs = _emit_attrs_from_lists(issues, latches, selects)
        if attrs:
            emits.append(UHIRControllerEmit(f"T{time_step}", attrs))

    return UHIRController(
        name=f"C_{header_region.id}",
        attributes={
            "encoding": encoding,
            "protocol": "act_done",
            "completion_order": "in_order",
            "overlap": True,
            "region": header_region.id,
            "parent_node": loop_node.id,
        },
        inputs=[
            UHIRPort("input", "act_valid", "i1"),
            UHIRPort("input", "done_ready", "i1"),
        ],
        outputs=[
            UHIRPort("output", "act_ready", "i1"),
            UHIRPort("output", "done_valid", "i1"),
        ],
        states=states,
        transitions=transitions,
        emits=emits,
    )


def _design_max_end(design: UHIRDesign) -> int:
    """Return the last occupied control step in one concrete bind design."""
    return max((node.attributes["end"] for region in design.regions for node in region.nodes), default=0)


def _emit_attrs_for_actions(actions: dict[str, tuple[str, ...]]) -> dict[str, object]:
    """Build one controller emit attribute bundle from one action dictionary."""
    return _emit_attrs_from_lists(
        list(actions.get("issue", ())),
        list(actions.get("latch", ())),
        list(actions.get("select", ())),
    )


def _emit_attrs_from_lists(
    issues: list[str],
    latches: list[str],
    selects: list[str],
) -> dict[str, object]:
    """Build one controller emit attribute bundle from normalized action lists."""
    attrs: dict[str, object] = {}
    if issues:
        attrs["issue"] = tuple(issues)
    if latches:
        attrs["latch"] = tuple(latches)
    if selects:
        attrs["select"] = tuple(selects)
    return attrs


def _concrete_subtree_max_end(region_id: str, region_by_id: dict[str, UHIRRegion]) -> int:
    """Return the last occupied concrete control step in one static region subtree."""
    max_end = 0

    def visit(current_region_id: str) -> None:
        nonlocal max_end
        region = region_by_id[current_region_id]
        for node in region.nodes:
            end = node.attributes.get("end")
            if not isinstance(end, int):
                raise ValueError(
                    f"dynamic call child controller currently requires concrete local callee timing; "
                    f"node '{region.id}/{node.id}' is missing integer end"
                )
            max_end = max(max_end, end)
        for node in region.nodes:
            for child_id in _node_children(node):
                visit(child_id)

    visit(region_id)
    return max_end


def _design_has_symbolic_timing(design: UHIRDesign) -> bool:
    """Return whether one bind artifact still carries symbolic timing."""
    for region in design.regions:
        if region.steps is not None and any(isinstance(value, TimingExpr) for value in region.steps):
            return True
        if isinstance(region.latency, TimingExpr) or isinstance(region.initiation_interval, TimingExpr):
            return True
        for node in region.nodes:
            for attr_name in ("delay", "start", "end", "ii"):
                if isinstance(node.attributes.get(attr_name), TimingExpr):
                    return True
    return False


def _state_code(index: int, encoding: str) -> int:
    """Return one concrete state code for one encoding family."""
    if encoding == "one_hot":
        return 1 << index
    return index


def _collect_time_step_actions(design: UHIRDesign) -> dict[int, dict[str, list[str]]]:
    """Collect issue/latch/select actions keyed by global control step."""
    actions: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"issue": [], "latch": [], "select": []})
    region_by_id = {region.id: region for region in design.regions}

    def append_region(region_id: str, offset: int, suffix: str = "") -> None:
        region = region_by_id[region_id]
        muxes_by_output: dict[str, list[UHIRMux]] = defaultdict(list)
        for mux in region.muxes:
            muxes_by_output[mux.output].append(mux)
        for node in region.nodes:
            bind = node.attributes.get("bind")
            class_name = node.attributes.get("class")
            if isinstance(bind, str) and isinstance(class_name, str) and class_name != "CTRL":
                start = node.attributes["start"] + offset
                actions[start]["issue"].append(f"{bind}<-{node.id}{suffix}")
        for binding in region.value_bindings:
            write_time = binding.live_intervals[0][0] + offset
            actions[write_time]["latch"].append(binding.register)
            for mux in muxes_by_output.get(binding.register, ()):
                actions[write_time]["select"].append(f"{mux.id}<-{mux.select}")

        for node in region.nodes:
            node_start = node.attributes.get("start")
            if not isinstance(node_start, int):
                continue
            if node.opcode == "loop":
                child_id = node.attributes.get("child")
                trip_count = node.attributes.get("static_trip_count")
                iter_ii = node.attributes.get("iter_initiation_interval")
                if not isinstance(child_id, str) or not isinstance(trip_count, int) or not isinstance(iter_ii, int):
                    raise ValueError(
                        f"fsm lowering currently requires fully static loop summaries; node '{region.id}/{node.id}' is missing child/static_trip_count/iter_initiation_interval"
                    )
                for iteration in range(trip_count):
                    append_loop_header(child_id, offset + node_start + iteration * iter_ii, iteration, expand_body=True)
                append_loop_header(child_id, offset + node_start + trip_count * iter_ii, trip_count, expand_body=False)
                continue
            for child_id in _node_children(node):
                append_region(child_id, offset, suffix=suffix)

    def append_loop_header(region_id: str, offset: int, iteration: int, *, expand_body: bool) -> None:
        region = region_by_id[region_id]
        suffix = f"@{iteration}"
        muxes_by_output: dict[str, list[UHIRMux]] = defaultdict(list)
        for mux in region.muxes:
            muxes_by_output[mux.output].append(mux)
        for node in region.nodes:
            bind = node.attributes.get("bind")
            class_name = node.attributes.get("class")
            if isinstance(bind, str) and isinstance(class_name, str) and class_name != "CTRL":
                start = node.attributes["start"] + offset
                actions[start]["issue"].append(f"{bind}<-{node.id}{suffix}")
        for binding in region.value_bindings:
            write_time = binding.live_intervals[0][0] + offset
            actions[write_time]["latch"].append(binding.register)
            for mux in muxes_by_output.get(binding.register, ()):
                actions[write_time]["select"].append(f"{mux.id}<-{mux.select}")

        for node in region.nodes:
            if node.opcode == "branch":
                true_child = node.attributes.get("true_child")
                false_child = node.attributes.get("false_child")
                if expand_body and isinstance(true_child, str):
                    append_region(true_child, offset, suffix=suffix)
                if not expand_body and isinstance(false_child, str):
                    append_region(false_child, offset, suffix=suffix)
                continue
            for child_id in _node_children(node):
                append_region(child_id, offset, suffix=suffix)

    for region in sorted((region.id for region in design.regions if region.parent is None)):
        append_region(region, 0)

    for kind_actions in actions.values():
        for key in ("issue", "latch", "select"):
            kind_actions[key].sort()
    return dict(actions)


def _node_children(node: UHIRNode) -> tuple[str, ...]:
    """Return direct child region ids referenced by one hierarchy node."""
    children: list[str] = []
    for attr_name in ("child", "true_child", "false_child"):
        child = node.attributes.get(attr_name)
        if isinstance(child, str) and child:
            children.append(child)
    return tuple(children)


def _collect_region_local_actions(region: UHIRRegion) -> dict[int, dict[str, tuple[str, ...]]]:
    """Collect local issue/latch/select actions for one region only."""
    actions: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"issue": [], "latch": [], "select": []})
    muxes_by_output: dict[str, list[UHIRMux]] = defaultdict(list)
    for mux in region.muxes:
        muxes_by_output[mux.output].append(mux)
    for node in region.nodes:
        bind = node.attributes.get("bind")
        class_name = node.attributes.get("class")
        start = node.attributes.get("start")
        if isinstance(bind, str) and isinstance(class_name, str) and class_name != "CTRL" and isinstance(start, int):
            actions[start]["issue"].append(f"{bind}<-{node.id}")
    for binding in region.value_bindings:
        if not binding.live_intervals:
            continue
        write_time = binding.live_intervals[0][0]
        actions[write_time]["latch"].append(binding.register)
        for mux in muxes_by_output.get(binding.register, ()):
            actions[write_time]["select"].append(f"{mux.id}<-{mux.select}")
    return {
        time_step: {
            key: tuple(sorted(values))
            for key, values in kind_actions.items()
            if values
        }
        for time_step, kind_actions in actions.items()
    }


def _collect_region_subtree_actions(region_id: str, region_by_id: dict[str, UHIRRegion]) -> dict[int, dict[str, tuple[str, ...]]]:
    """Collect issue/latch/select actions for one static subtree rooted at one child region."""
    actions: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"issue": [], "latch": [], "select": []})

    def visit(current_region_id: str) -> None:
        region = region_by_id[current_region_id]
        muxes_by_output: dict[str, list[UHIRMux]] = defaultdict(list)
        for mux in region.muxes:
            muxes_by_output[mux.output].append(mux)
        for node in region.nodes:
            bind = node.attributes.get("bind")
            class_name = node.attributes.get("class")
            start = node.attributes.get("start")
            if isinstance(bind, str) and isinstance(class_name, str) and class_name != "CTRL" and isinstance(start, int):
                actions[start]["issue"].append(f"{bind}<-{node.id}")
        for binding in region.value_bindings:
            if not binding.live_intervals:
                continue
            write_time = binding.live_intervals[0][0]
            actions[write_time]["latch"].append(binding.register)
            for mux in muxes_by_output.get(binding.register, ()):
                actions[write_time]["select"].append(f"{mux.id}<-{mux.select}")
        for node in region.nodes:
            for child_id in _node_children(node):
                visit(child_id)

    visit(region_id)
    return {
        time_step: {
            key: tuple(sorted(values))
            for key, values in kind_actions.items()
            if values
        }
        for time_step, kind_actions in actions.items()
    }


def _collect_dynamic_phases(region: UHIRRegion) -> dict[int, tuple[UHIRNode, ...]]:
    """Group one top-level region's local nodes into coarse dynamic controller phases."""
    node_by_id = {node.id: node for node in region.nodes}

    def neighbors(node: UHIRNode) -> tuple[UHIRNode, ...]:
        next_nodes: list[UHIRNode] = []
        for edge in region.edges:
            if not edge.directed:
                continue
            target = node_by_id.get(edge.target)
            if edge.source == node.id and target is not None:
                next_nodes.append(target)
        return tuple(next_nodes)

    ordered_nodes = topological_sort(region.nodes, neighbors, key=lambda node: node.id)
    phase_of: dict[str, int] = {}
    preds_by_id: dict[str, list[str]] = defaultdict(list)
    for edge in region.edges:
        if edge.directed and edge.source in node_by_id and edge.target in node_by_id:
            preds_by_id[edge.target].append(edge.source)

    for node in ordered_nodes:
        preds = preds_by_id.get(node.id, [])
        phase_of[node.id] = max((phase_of[pred_id] + _phase_advance(node_by_id[pred_id]) for pred_id in preds), default=0)

    phases: dict[int, list[UHIRNode]] = defaultdict(list)
    for node in ordered_nodes:
        if node.opcode == "nop":
            continue
        phases[phase_of[node.id]].append(node)
    return {phase_id: tuple(nodes) for phase_id, nodes in sorted(phases.items())}


def _phase_advance(node: UHIRNode) -> int:
    """Return the coarse phase distance contributed by one predecessor node."""
    delay = node.attributes.get("delay")
    if _is_symbolic_hierarchy_node(node):
        return 1
    if isinstance(delay, int):
        return max(delay, 0)
    return 1


def _is_symbolic_hierarchy_node(node: UHIRNode) -> bool:
    """Return whether one node is controlled through the dynamic ready/done contract."""
    return (
        node.opcode in {"call", "loop", "branch"}
        and node.attributes.get("timing") == "symbolic"
        and isinstance(node.attributes.get("completion"), str)
    )


def _dynamic_phase_ready_condition(nodes: tuple[UHIRNode, ...]) -> str | None:
    """Return the admission condition for one dynamic phase."""
    ready_conditions = sorted(
        ready
        for node in nodes
        if _is_symbolic_hierarchy_node(node)
        and isinstance((ready := node.attributes.get("ready")), str)
    )
    return _conjoin_conditions(ready_conditions)


def _dynamic_phase_completion_condition(nodes: list[UHIRNode]) -> str | None:
    """Return the completion condition for one symbolic wait state."""
    completion_conditions = sorted(
        completion
        for node in nodes
        if isinstance((completion := node.attributes.get("completion")), str)
    )
    return _conjoin_conditions(completion_conditions)


def _conjoin_conditions(parts: tuple[str | None, ...] | list[str] | tuple[str, ...]) -> str | None:
    """Join condition fragments with && while skipping missing parts."""
    normalized = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    if not normalized:
        return None
    return " && ".join(normalized)


def _require_loop_condition(loop_node: UHIRNode, attr_name: str) -> str:
    """Return one required dynamic loop control condition."""
    condition = loop_node.attributes.get(attr_name)
    if not isinstance(condition, str) or not condition:
        raise ValueError(f"dynamic loop node '{loop_node.id}' is missing {attr_name}=...")
    return condition


def _clone_fsm_region(region: UHIRRegion) -> UHIRRegion:
    cloned = UHIRRegion(id=region.id, kind=region.kind, parent=region.parent)
    cloned.region_refs = [UHIRRegionRef(ref.target) for ref in region.region_refs]
    cloned.nodes = [_clone_node(node) for node in region.nodes]
    cloned.edges = [UHIREdge(edge.kind, edge.source, edge.target, dict(edge.attributes), edge.directed) for edge in region.edges]
    cloned.mappings = [UHIRSourceMap(mapping.node_id, mapping.source_id) for mapping in region.mappings]
    cloned.value_bindings = [
        UHIRValueBinding(binding.producer, binding.register, binding.live_intervals)
        for binding in region.value_bindings
    ]
    cloned.muxes = [UHIRMux(mux.id, mux.inputs, mux.output, mux.select, dict(mux.attributes)) for mux in region.muxes]
    cloned.steps = region.steps
    cloned.latency = region.latency
    cloned.initiation_interval = region.initiation_interval
    return cloned


def _clone_node(node: UHIRNode) -> UHIRNode:
    return UHIRNode(node.id, node.opcode, node.operands, node.result_type, dict(node.attributes))


def _clone_port(port: UHIRPort) -> UHIRPort:
    return UHIRPort(port.direction, port.name, port.type)


def _clone_constant(constant: UHIRConstant) -> UHIRConstant:
    return UHIRConstant(constant.name, constant.value, constant.type)
