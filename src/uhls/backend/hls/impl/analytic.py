"""Analytical HLS-side PPA estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
import re
from pathlib import Path
from typing import Iterable, Mapping

from uhls.backend.hls.lib import resolve_component_definition, resolve_component_ppa_estimate


_REG_BIT_AREA_UM2 = 1.0
_LOCAL_MEM_BIT_AREA_UM2 = 1.2
_MUX_INPUT_BIT_AREA_UM2 = 0.25
_CONTROLLER_STATE_BIT_AREA_UM2 = 1.0
_CONTROLLER_STATE_LOGIC_AREA_UM2 = 2.0
_CONTROLLER_TRANSITION_AREA_UM2 = 0.5
_CONTROLLER_EMIT_AREA_UM2 = 0.5
_FALLBACK_COMPONENT_AREA_UM2 = 64.0


@dataclass(frozen=True)
class AnalyticalAreaItem:
    """One analytical area contribution line."""

    category: str
    label: str
    total_area_um2: float
    count: int | None = None
    unit_area_um2: float | None = None
    detail: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class AnalyticalAreaReport:
    """One analytical area estimate."""

    design_name: str
    stage: str
    total_area_um2: float
    items: tuple[AnalyticalAreaItem, ...]
    warnings: tuple[str, ...] = ()


def estimate_analytical_area(
    design,
    *,
    component_library: Mapping[str, Mapping[str, object]] | None = None,
    library_root: Path | None = None,
) -> AnalyticalAreaReport:
    """Estimate area directly from one bind/fsm/µglIR design."""
    stage = str(getattr(design, "stage", "")).strip().lower()
    if stage not in {"bind", "fsm", "uglir"}:
        raise ValueError(f"analytical area estimation expects bind/fsm/uglir input, got stage '{stage}'")

    items: list[AnalyticalAreaItem] = []
    warnings: list[str] = []
    component_items, component_warnings = _estimate_component_resources(
        getattr(design, "resources", ()),
        component_library=component_library,
        library_root=library_root,
    )
    items.extend(component_items)
    warnings.extend(component_warnings)

    reg_item = _estimate_register_resources(getattr(design, "resources", ()))
    if reg_item is not None:
        items.append(reg_item)
    mem_item = _estimate_local_mem_resources(getattr(design, "resources", ()))
    if mem_item is not None:
        items.append(mem_item)

    if stage in {"bind", "fsm"}:
        mux_item = _estimate_uhir_muxes(getattr(design, "regions", ()), getattr(design, "resources", ()))
        if mux_item is not None:
            items.append(mux_item)
    else:
        mux_item = _estimate_uglir_muxes(getattr(design, "muxes", ()))
        if mux_item is not None:
            items.append(mux_item)

    if stage == "fsm":
        controller_item = _estimate_controllers(getattr(design, "controllers", ()))
        if controller_item is not None:
            items.append(controller_item)

    total_area = sum(item.total_area_um2 for item in items)
    return AnalyticalAreaReport(
        design_name=str(getattr(design, "name", "<unnamed>")),
        stage=stage,
        total_area_um2=total_area,
        items=tuple(sorted(items, key=lambda item: (-item.total_area_um2, item.category, item.label))),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _estimate_component_resources(
    resources: Iterable[object],
    *,
    component_library: Mapping[str, Mapping[str, object]] | None,
    library_root: Path | None,
) -> tuple[list[AnalyticalAreaItem], list[str]]:
    grouped: dict[tuple[str, str, str], AnalyticalAreaItem] = {}
    warnings: list[str] = []
    for resource in resources:
        kind = str(getattr(resource, "kind", ""))
        if kind not in {"fu", "inst"}:
            continue
        component_spec = getattr(resource, "target", None) or getattr(resource, "value", None)
        if not isinstance(component_spec, str) or not component_spec.strip():
            continue
        try:
            unit_area, source = _estimate_component_unit_area(
                component_spec.strip(),
                component_library=component_library,
                library_root=library_root,
            )
        except ValueError as exc:
            raise ValueError(f"failed to estimate component '{component_spec}': {exc}") from exc
        if source == "fallback":
            warnings.append(
                f"component '{component_spec}' is missing ppa_estimate.area_um2 and LEF area; using fallback heuristic"
            )
        key = ("component", component_spec.strip(), source)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = AnalyticalAreaItem(
                category="component",
                label=component_spec.strip(),
                count=1,
                unit_area_um2=unit_area,
                total_area_um2=unit_area,
                source=source,
            )
            continue
        grouped[key] = AnalyticalAreaItem(
            category=existing.category,
            label=existing.label,
            count=(existing.count or 0) + 1,
            unit_area_um2=existing.unit_area_um2,
            total_area_um2=existing.total_area_um2 + unit_area,
            source=existing.source,
        )
    return list(grouped.values()), warnings


def _estimate_component_unit_area(
    component_spec: str,
    *,
    component_library: Mapping[str, Mapping[str, object]] | None,
    library_root: Path | None,
) -> tuple[float, str]:
    if component_library is None:
        return _FALLBACK_COMPONENT_AREA_UM2, "fallback"
    area_um2 = resolve_component_ppa_estimate(dict(component_library), component_spec, "area_um2")
    if area_um2 is not None:
        return area_um2, "ppa_estimate"
    _base_name, _params, component = resolve_component_definition(dict(component_library), component_spec)
    hdl = component.get("hdl")
    if isinstance(hdl, Mapping):
        lef_files = hdl.get("lef_files")
        if isinstance(lef_files, list):
            for lef_path in lef_files:
                if isinstance(lef_path, str) and lef_path.strip():
                    resolved = _resolve_collateral_path(lef_path, library_root)
                    if resolved.exists():
                        area = _parse_lef_area_um2(resolved)
                        if area is not None:
                            return area, "lef"
    return _FALLBACK_COMPONENT_AREA_UM2, "fallback"


def _resolve_collateral_path(path_text: str, library_root: Path | None) -> Path:
    path = Path(path_text)
    if path.is_absolute() or library_root is None:
        return path
    return (library_root / path).resolve()


def _parse_lef_area_um2(path: Path) -> float | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(r"^\s*SIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)\s*;\s*$", text, re.MULTILINE)
    if match is None:
        return None
    return float(match.group(1)) * float(match.group(2))


def _estimate_register_resources(resources: Iterable[object]) -> AnalyticalAreaItem | None:
    total_bits = 0
    count = 0
    for resource in resources:
        if str(getattr(resource, "kind", "")) != "reg":
            continue
        total_bits += _safe_type_bits(str(getattr(resource, "value", "")))
        count += 1
    if count == 0:
        return None
    return AnalyticalAreaItem(
        category="registers",
        label="register file",
        count=count,
        total_area_um2=total_bits * _REG_BIT_AREA_UM2,
        detail=f"bits={total_bits}",
        source="heuristic",
    )


def _estimate_local_mem_resources(resources: Iterable[object]) -> AnalyticalAreaItem | None:
    total_bits = 0
    count = 0
    for resource in resources:
        if str(getattr(resource, "kind", "")) != "mem":
            continue
        total_bits += _safe_type_bits(str(getattr(resource, "value", "")))
        count += 1
    if count == 0:
        return None
    return AnalyticalAreaItem(
        category="memories",
        label="local memories",
        count=count,
        total_area_um2=total_bits * _LOCAL_MEM_BIT_AREA_UM2,
        detail=f"bits={total_bits}",
        source="heuristic",
    )


def _estimate_uhir_muxes(regions: Iterable[object], resources: Iterable[object]) -> AnalyticalAreaItem | None:
    reg_widths = {
        str(getattr(resource, "id", "")): _safe_type_bits(str(getattr(resource, "value", "")))
        for resource in resources
        if str(getattr(resource, "kind", "")) == "reg"
    }
    count = 0
    weighted_inputs = 0
    total_area = 0.0
    for region in regions:
        for mux in getattr(region, "muxes", ()):
            inputs = tuple(getattr(mux, "inputs", ()))
            fanin = max(len(inputs) - 1, 1)
            width = reg_widths.get(str(getattr(mux, "output", "")), 1)
            total_area += width * fanin * _MUX_INPUT_BIT_AREA_UM2
            count += 1
            weighted_inputs += fanin
    if count == 0:
        return None
    return AnalyticalAreaItem(
        category="muxes",
        label="datapath muxes",
        count=count,
        total_area_um2=total_area,
        detail=f"weighted_inputs={weighted_inputs}",
        source="heuristic",
    )


def _estimate_uglir_muxes(muxes: Iterable[object]) -> AnalyticalAreaItem | None:
    count = 0
    weighted_inputs = 0
    total_area = 0.0
    for mux in muxes:
        fanin = max(len(getattr(mux, "cases", ())) - 1, 1)
        width = _safe_type_bits(str(getattr(mux, "type", "")))
        total_area += width * fanin * _MUX_INPUT_BIT_AREA_UM2
        count += 1
        weighted_inputs += fanin
    if count == 0:
        return None
    return AnalyticalAreaItem(
        category="muxes",
        label="explicit muxes",
        count=count,
        total_area_um2=total_area,
        detail=f"weighted_inputs={weighted_inputs}",
        source="heuristic",
    )


def _estimate_controllers(controllers: Iterable[object]) -> AnalyticalAreaItem | None:
    controller_list = list(controllers)
    if not controller_list:
        return None
    state_bits = 0
    transition_count = 0
    emit_count = 0
    for controller in controller_list:
        states = list(getattr(controller, "states", ()))
        encoding = str(getattr(controller, "attributes", {}).get("encoding", "binary"))
        state_bits += _controller_state_bits(len(states), encoding)
        transition_count += len(getattr(controller, "transitions", ()))
        emit_count += len(getattr(controller, "emits", ()))
    total_area = (
        state_bits * _CONTROLLER_STATE_BIT_AREA_UM2
        + sum(len(getattr(controller, "states", ())) for controller in controller_list) * _CONTROLLER_STATE_LOGIC_AREA_UM2
        + transition_count * _CONTROLLER_TRANSITION_AREA_UM2
        + emit_count * _CONTROLLER_EMIT_AREA_UM2
    )
    return AnalyticalAreaItem(
        category="controllers",
        label="controllers",
        count=len(controller_list),
        total_area_um2=total_area,
        detail=f"state_bits={state_bits} transitions={transition_count} emits={emit_count}",
        source="heuristic",
    )


def _controller_state_bits(num_states: int, encoding: str) -> int:
    if num_states <= 0:
        return 0
    if encoding == "one_hot":
        return num_states
    return max(1, ceil(log2(num_states)))


def _safe_type_bits(type_name: str) -> int:
    try:
        return _type_bits(type_name)
    except ValueError:
        return 1


def _type_bits(type_name: str) -> int:
    text = type_name.strip()
    if not text:
        raise ValueError("empty type")
    if text in {"clock", "reset", "i1", "u1"}:
        return 1
    if array_match := re.fullmatch(r"(.+)\[(\d+)\]", text):
        base_type, length = array_match.groups()
        return _type_bits(base_type.strip()) * int(length)
    if scalar_match := re.fullmatch(r"[iu](\d+)", text):
        return int(scalar_match.group(1))
    raise ValueError(f"unsupported type '{type_name}'")


__all__ = [
    "AnalyticalAreaItem",
    "AnalyticalAreaReport",
    "estimate_analytical_area",
]
