"""Validation for uglir designs prior to RTL emission."""

from __future__ import annotations

from collections import Counter, defaultdict
import re

from .model import UGLIRDesign

_TYPED_INT_RE = re.compile(r"(?<![\w$])(-?\d+):(i|u)(\d+)\b")
_VERILOG_INT_RE = re.compile(r"(?<![\w$])\d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ]+")
_IDENT_RE = re.compile(r"\b[A-Za-z_][\w$]*\b")


def validate_uglir_for_rtl(design: UGLIRDesign) -> None:
    """Validate one uglir design against the current RTL backend contract."""
    if design.stage != "uglir":
        raise ValueError(f"uglir validator expects uglir input, got stage '{design.stage}'")

    _validate_unique_signal_names(design)

    inputs = {port.name: port.type for port in design.inputs}
    outputs = {port.name: port.type for port in design.outputs}
    constants = {const.name: const.type for const in design.constants}
    resources = {resource.id: resource for resource in design.resources}
    resource_kinds = {resource.id: resource.kind for resource in design.resources}
    signal_types = dict(inputs)
    signal_types.update(outputs)
    signal_types.update(constants)
    signal_types.update(
        {
            resource.id: (_mem_element_type(resource.value) if resource.kind == "mem" else resource.value)
            for resource in design.resources
            if resource.kind in {"reg", "net", "mux", "mem"}
        }
    )
    instance_ids = {resource.id for resource in design.resources if resource.kind == "inst"}
    mux_resource_types = {resource.id: resource.value for resource in design.resources if resource.kind == "mux"}
    assign_targets = Counter(assign.target for assign in design.assigns)
    ctrl_labels = _ctrl_labels_by_signal(design)

    for attachment in design.attachments:
        if attachment.instance not in instance_ids:
            raise ValueError(f"uglir attachment '{attachment.instance}.{attachment.port}(...)' references unknown instance")
        if attachment.signal not in signal_types:
            raise ValueError(f"uglir attachment '{attachment.instance}.{attachment.port}({attachment.signal})' references unknown signal")

    attachment_ports: dict[str, set[str]] = defaultdict(set)
    for attachment in design.attachments:
        if attachment.port in attachment_ports[attachment.instance]:
            raise ValueError(
                f"uglir instance '{attachment.instance}' must not attach port '{attachment.port}' more than once"
            )
        attachment_ports[attachment.instance].add(attachment.port)

    for assign in design.assigns:
        target_kind = resource_kinds.get(assign.target)
        if assign.target not in outputs and target_kind != "net":
            raise ValueError(
                f"uglir assign '{assign.target} = ...' must target an output port or net resource"
            )
        _validate_expr_identifiers(assign.expr, assign.target, signal_types, ctrl_labels)
        source_type = _infer_simple_expr_type(assign.expr, signal_types, constants)
        target_type = signal_types[assign.target]
        if source_type is not None and not _rtl_types_compatible(target_type, source_type):
            raise ValueError(
                f"uglir assign '{assign.target} = ...' has incompatible type '{source_type}' for target '{target_type}'"
            )

    for output_name in outputs:
        if assign_targets[output_name] != 1:
            raise ValueError(f"uglir output '{output_name}' must have exactly one assign driver")

    for resource in design.resources:
        if resource.kind == "net" and resource.value == "ctrl" and assign_targets[resource.id] != 1:
            raise ValueError(f"uglir ctrl net '{resource.id}' must have exactly one assign driver")

    muxes = {mux.name: mux for mux in design.muxes}
    if len(muxes) != len(design.muxes):
        raise ValueError("uglir mux names must be unique")
    for mux_name in mux_resource_types:
        if mux_name not in muxes:
            raise ValueError(f"uglir mux resource '{mux_name}' must have a matching mux declaration")
    for mux in design.muxes:
        if mux.name not in mux_resource_types:
            raise ValueError(f"uglir mux '{mux.name}' must have a declared mux resource")
        if mux.type != mux_resource_types[mux.name]:
            raise ValueError(
                f"uglir mux '{mux.name}' type '{mux.type}' does not match resource type '{mux_resource_types[mux.name]}'"
            )
        if mux.select not in signal_types:
            raise ValueError(f"uglir mux '{mux.name}' references unknown select signal '{mux.select}'")
        if signal_types[mux.select] != "ctrl":
            raise ValueError(f"uglir mux '{mux.name}' select signal '{mux.select}' must have type ctrl")
        if not mux.cases:
            raise ValueError(f"uglir mux '{mux.name}' must declare at least one case")
        case_keys = [case.key for case in mux.cases]
        if len(case_keys) != len(set(case_keys)):
            raise ValueError(f"uglir mux '{mux.name}' must use unique case keys")
        for case in mux.cases:
            if case.source not in signal_types:
                raise ValueError(f"uglir mux '{mux.name}' case '{case.key}' references unknown source '{case.source}'")
            if not _rtl_types_compatible(mux.type, signal_types[case.source]):
                raise ValueError(
                    f"uglir mux '{mux.name}' case '{case.key}' source '{case.source}' has type '{signal_types[case.source]}', expected '{mux.type}'"
                )

    reg_seq_drivers = Counter()
    reg_seq_owners: dict[str, int] = {}
    for seq_index, seq_block in enumerate(design.seq_blocks):
        if seq_block.clock not in inputs:
            raise ValueError(f"uglir seq block clock '{seq_block.clock}' must be an input port")
        if inputs[seq_block.clock] != "clock":
            raise ValueError(f"uglir seq block clock '{seq_block.clock}' must have type clock")
        if seq_block.reset is not None:
            _validate_expr_identifiers(seq_block.reset, None, signal_types, ctrl_labels)
        reset_targets = set()
        for update in seq_block.reset_updates:
            target_base, target_index = _split_seq_target(update.target)
            if target_base not in resources or resources[target_base].kind not in {"reg", "mem"}:
                raise ValueError(f"uglir sequential update target '{update.target}' must be a reg or mem resource")
            if resources[target_base].kind == "reg" and target_index is not None:
                raise ValueError(f"uglir reg target '{update.target}' must not use indexed syntax")
            if update.target in reset_targets:
                raise ValueError(f"uglir seq block reset must not update reg '{update.target}' more than once")
            reset_targets.add(update.target)
            if resources[target_base].kind == "reg":
                owner = reg_seq_owners.setdefault(target_base, seq_index)
                if owner != seq_index:
                    raise ValueError(f"uglir reg '{target_base}' must not be driven by multiple sequential blocks")
                reg_seq_drivers[target_base] += 1
            if target_index is not None:
                _validate_expr_identifiers(target_index, None, signal_types, ctrl_labels)
            _validate_expr_identifiers(update.value, None, signal_types, ctrl_labels)
            value_type = _infer_simple_expr_type(update.value, signal_types, constants)
            target_type = signal_types[target_base]
            if value_type is not None and not _rtl_types_compatible(target_type, value_type):
                raise ValueError(
                    f"uglir sequential update '{update.target} <= ...' has incompatible type '{value_type}' for target '{target_type}'"
                )

        update_targets = set()
        for update in seq_block.updates:
            target_base, target_index = _split_seq_target(update.target)
            if target_base not in resources or resources[target_base].kind not in {"reg", "mem"}:
                raise ValueError(f"uglir sequential update target '{update.target}' must be a reg or mem resource")
            if resources[target_base].kind == "reg" and target_index is not None:
                raise ValueError(f"uglir reg target '{update.target}' must not use indexed syntax")
            if update.target in update_targets:
                raise ValueError(f"uglir seq block must not update reg '{update.target}' more than once")
            update_targets.add(update.target)
            if resources[target_base].kind == "reg":
                owner = reg_seq_owners.setdefault(target_base, seq_index)
                if owner != seq_index:
                    raise ValueError(f"uglir reg '{target_base}' must not be driven by multiple sequential blocks")
                reg_seq_drivers[target_base] += 1
            if target_index is not None:
                _validate_expr_identifiers(target_index, None, signal_types, ctrl_labels)
            _validate_expr_identifiers(update.value, None, signal_types, ctrl_labels)
            value_type = _infer_simple_expr_type(update.value, signal_types, constants)
            target_type = signal_types[target_base]
            if value_type is not None and not _rtl_types_compatible(target_type, value_type):
                raise ValueError(
                    f"uglir sequential update '{update.target} <= ...' has incompatible type '{value_type}' for target '{target_type}'"
                )
            if update.enable is not None:
                _validate_expr_identifiers(update.enable, None, signal_types, ctrl_labels)

    for resource in design.resources:
        if resource.kind == "reg":
            if assign_targets[resource.id] != 0:
                raise ValueError(f"uglir reg '{resource.id}' must not be driven by a continuous assign")
            if reg_seq_drivers[resource.id] == 0:
                raise ValueError(f"uglir reg '{resource.id}' must be driven by a sequential block")
        if resource.kind == "mem":
            _mem_element_type(resource.value)


def _validate_unique_signal_names(design: UHIRDesign) -> None:
    names = [port.name for port in design.inputs]
    names.extend(port.name for port in design.outputs)
    names.extend(const.name for const in design.constants)
    names.extend(resource.id for resource in design.resources)
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"uglir signal/resource names must be unique: {', '.join(sorted(duplicates))}")


def _ctrl_labels_by_signal(design: UHIRDesign) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    for mux in design.muxes:
        labels[mux.select] = {case.key for case in mux.cases}
    return labels


def _validate_expr_identifiers(
    expr: str,
    target: str | None,
    signal_types: dict[str, str],
    ctrl_labels: dict[str, set[str]],
) -> None:
    allowed = set(signal_types)
    allowed.update({"true", "false"})
    if target is not None and signal_types.get(target) == "ctrl":
        allowed.update(ctrl_labels.get(target, set()))
    for ident in _expr_identifiers(expr):
        if ident not in allowed:
            raise ValueError(f"uglir expression '{expr}' references unknown symbol '{ident}'")


def _expr_identifiers(expr: str) -> set[str]:
    stripped = _TYPED_INT_RE.sub("", expr)
    stripped = _VERILOG_INT_RE.sub("", stripped)
    return {token for token in _IDENT_RE.findall(stripped) if token not in {"s", "d"}}


def _infer_simple_expr_type(
    expr: str,
    signal_types: dict[str, str],
    constant_types: dict[str, str],
) -> str | None:
    text = expr.strip()
    if text in signal_types:
        return signal_types[text]
    if text in constant_types:
        return constant_types[text]
    if text in {"true", "false"}:
        return "i1"
    match = re.fullmatch(r"(-?\d+):(i|u)(\d+)", text)
    if match is not None:
        return f"{match.group(2)}{match.group(3)}"
    return None


def _rtl_types_compatible(target_type: str, source_type: str) -> bool:
    if target_type == source_type:
        return True
    if _is_scalar_int_type(target_type) and _is_scalar_int_type(source_type):
        return True
    return False


def _is_scalar_int_type(type_name: str) -> bool:
    return re.fullmatch(r"[iu]\d+", type_name) is not None


def _split_seq_target(target: str) -> tuple[str, str | None]:
    match = re.fullmatch(r"([_A-Za-z][\w$]*)(?:\[(.+)\])?", target)
    if match is None:
        raise ValueError(f"uglir sequential update target '{target}' must be an identifier or indexed memory element")
    return match.group(1), match.group(2)


def _mem_element_type(type_name: str) -> str:
    match = re.fullmatch(r"([iu]\d+)\[(\d+)\]", type_name)
    if match is None:
        raise ValueError(f"uglir memory resource type '{type_name}' must be <scalar>[<depth>]")
    depth = int(match.group(2))
    if depth <= 0:
        raise ValueError(f"uglir memory resource type '{type_name}' must use depth > 0")
    return match.group(1)
