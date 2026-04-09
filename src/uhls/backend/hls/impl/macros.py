"""Target-dispatched macro-collateral helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from uhls.backend.hls.lib import resolve_component_definition
from uhls.backend.hls.uglir import UGLIRDesign


@dataclass(slots=True, frozen=True)
class MacroCollateral:
    """One macro instance/collateral bundle needed by backend ASIC flows."""

    instance_name: str
    component_name: str
    module_name: str
    verilog_sources: tuple[str, ...] = ()
    include_dirs: tuple[str, ...] = ()
    defines: tuple[str, ...] = ()
    lef_files: tuple[str, ...] = ()
    liberty_files: tuple[str, ...] = ()
    gds_files: tuple[str, ...] = ()


def collect_flow_macros(
    target: str | None,
    design: UGLIRDesign,
    component_library: Mapping[str, Mapping[str, object]],
) -> tuple[MacroCollateral, ...]:
    """Collect backend macro collateral for one target from one µglIR design."""
    normalized = _normalize_target(target)
    if normalized == "ihp130":
        from .vendor.ihp130.macros import collect_ihp130_macros

        return _merge_macro_collaterals(
            _collect_generic_macros(design, component_library),
            collect_ihp130_macros(design, component_library),
        )
    return _collect_generic_macros(design, component_library)


def _collect_generic_macros(
    design: UGLIRDesign,
    component_library: Mapping[str, Mapping[str, object]],
) -> tuple[MacroCollateral, ...]:
    collaterals: list[MacroCollateral] = []
    for resource in design.resources:
        if resource.kind != "inst":
            continue
        component_spec = resource.target if resource.target is not None else resource.value
        _base_name, _params, component = resolve_component_definition(dict(component_library), component_spec)
        hdl = component.get("hdl")
        if not isinstance(hdl, Mapping):
            continue
        language = str(hdl.get("language", "")).strip().lower()
        module_name = str(hdl.get("module", "")).strip()
        if language != "verilog" or not module_name:
            continue
        sources = tuple(_collect_string_list(hdl, "sources") or _collect_optional_single_string(hdl, "source"))
        include_dirs = tuple(_collect_string_list(hdl, "include_dirs"))
        defines = tuple(_collect_string_list(hdl, "defines"))
        lef_files = tuple(_collect_string_list(hdl, "lef_files"))
        liberty_files = tuple(_collect_string_list(hdl, "liberty_files"))
        gds_files = tuple(_collect_string_list(hdl, "gds_files"))
        if not sources:
            continue
        collaterals.append(
            MacroCollateral(
                instance_name=resource.id,
                component_name=str(component_spec),
                module_name=module_name,
                verilog_sources=sources,
                include_dirs=include_dirs,
                defines=defines,
                lef_files=lef_files,
                liberty_files=liberty_files,
                gds_files=gds_files,
            )
        )
    return tuple(collaterals)


def _collect_string_list(payload: Mapping[str, object], key: str) -> list[str]:
    raw = payload.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [str(entry).strip() for entry in raw if str(entry).strip()]


def _collect_optional_single_string(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    raw = payload.get(key)
    if isinstance(raw, str) and raw.strip():
        return (raw.strip(),)
    return ()


def _merge_macro_collaterals(
    base: tuple[MacroCollateral, ...],
    overrides: tuple[MacroCollateral, ...],
) -> tuple[MacroCollateral, ...]:
    merged: list[MacroCollateral] = []
    by_instance: dict[str, MacroCollateral] = {collateral.instance_name: collateral for collateral in overrides}
    seen: set[str] = set()
    for collateral in base:
        override = by_instance.get(collateral.instance_name)
        chosen = override if override is not None else collateral
        merged.append(chosen)
        seen.add(chosen.instance_name)
    for collateral in overrides:
        if collateral.instance_name not in seen:
            merged.append(collateral)
            seen.add(collateral.instance_name)
    return tuple(merged)


def _normalize_target(target: str | None) -> str:
    return "" if target is None else target.strip().lower()


__all__ = ["MacroCollateral", "collect_flow_macros"]
