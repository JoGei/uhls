"""DOT rendering helpers for fsm-stage controllers."""

from __future__ import annotations

from uhls.utils.dot import escape_dot_label as _escape

from uhls.backend.hls.uhir.model import UHIRController, UHIRDesign


def fsm_to_dot(design: UHIRDesign) -> str:
    """Render one fsm-stage design as a controller state diagram."""
    lines = [f'digraph "{design.name}.fsm" {{', "  rankdir=TB;", '  labelloc="t";', f'  label="{_escape(design.name)} FSM";']
    for controller in design.controllers:
        lines.extend(_render_controller(controller))
    lines.append("}")
    return "\n".join(lines)


def _render_controller(controller: UHIRController) -> list[str]:
    lines = [f'  subgraph "cluster_{controller.name}" {{']
    lines.append(f'    label="{_escape(_controller_label(controller))}";')
    lines.append("    color=gray70;")
    emit_by_state = {emit.state: emit.attributes for emit in controller.emits}
    for state in controller.states:
        shape = "doublecircle" if state.name == "DONE" else "box" if state.name == "IDLE" else "ellipse"
        fillcolor = "#e6f2ff" if state.name == "IDLE" else "#e8f8e8" if state.name == "DONE" else "#ffffff"
        label = _state_label(state.name, state.attributes, emit_by_state.get(state.name))
        lines.append(
            f'    "{controller.name}:{state.name}" [label="{_escape(label)}", shape={shape}, style=filled, fillcolor="{fillcolor}"];'
        )
    lines.append(f'    "{controller.name}:entry" [label="", shape=point, width=0.12];')
    lines.append(f'    "{controller.name}:entry" -> "{controller.name}:IDLE";')
    for transition in controller.transitions:
        attrs: list[str] = ['color="#4c78a8"']
        when = transition.attributes.get("when")
        if isinstance(when, str) and when:
            attrs.append(f'label="{_escape(when)}"')
            attrs.append('fontcolor="#4c78a8"')
        lines.append(
            f'    "{controller.name}:{transition.source}" -> "{controller.name}:{transition.target}" [{", ".join(attrs)}];'
        )
    lines.append("  }")
    return lines


def _controller_label(controller: UHIRController) -> str:
    encoding = controller.attributes.get("encoding", "")
    protocol = controller.attributes.get("protocol", "")
    return f"{controller.name} ({encoding}, {protocol})"


def _state_label(
    state_name: str,
    state_attrs: dict[str, object],
    emit_attrs: dict[str, object] | None,
) -> str:
    lines = [state_name]
    code = state_attrs.get("code")
    if code is not None:
        lines.append(f"code={code}")
    if emit_attrs:
        for attr_name in sorted(emit_attrs):
            lines.append(f"{attr_name}={_format_emit_value(emit_attrs[attr_name])}")
    return "\n".join(lines)


def _format_emit_value(value: object) -> str:
    if isinstance(value, tuple):
        return "[" + ", ".join(str(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
