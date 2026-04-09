"""IHP130 macro-collateral helpers."""

from __future__ import annotations

from collections.abc import Mapping

from uhls.backend.hls.impl.macros import MacroCollateral
from uhls.backend.hls.lib import resolve_component_definition
from uhls.backend.hls.uglir import UGLIRDesign


def collect_ihp130_macros(
    design: UGLIRDesign,
    component_library: Mapping[str, Mapping[str, object]],
) -> tuple[MacroCollateral, ...]:
    """Collect instantiated IHP130 macros from one µglIR design."""
    collaterals: list[MacroCollateral] = []
    for resource in design.resources:
        if resource.kind != "inst":
            continue
        component_spec = resource.target if resource.target is not None else resource.value
        base_name, _params, component = resolve_component_definition(dict(component_library), component_spec)
        hdl = component.get("hdl")
        if not isinstance(hdl, Mapping):
            continue
        module_name = str(hdl.get("module", "")).strip()
        if not module_name.startswith("RM_IHPSG13_"):
            continue
        sources = _collect_sources(hdl)
        include_dirs = _collect_string_list(hdl, "include_dirs")
        defines = _collect_string_list(hdl, "defines")
        lef_files = _collect_string_list(hdl, "lef_files")
        liberty_files = _collect_string_list(hdl, "liberty_files")
        gds_files = _collect_string_list(hdl, "gds_files")
        collaterals.append(
            MacroCollateral(
                instance_name=resource.id,
                component_name=base_name,
                module_name=module_name,
                verilog_sources=tuple(sources),
                include_dirs=tuple(include_dirs),
                defines=tuple(defines),
                lef_files=tuple(lef_files),
                liberty_files=tuple(liberty_files),
                gds_files=tuple(gds_files),
            )
        )
    return tuple(collaterals)


def _collect_sources(hdl: Mapping[str, object]) -> list[str]:
    sources = _collect_string_list(hdl, "sources")
    if sources:
        return sources
    source = hdl.get("source")
    if isinstance(source, str) and source.strip():
        return [source.strip()]
    return []


def _collect_string_list(payload: Mapping[str, object], key: str) -> list[str]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        return []
    return [str(entry).strip() for entry in raw if str(entry).strip()]


__all__ = ["collect_ihp130_macros"]
