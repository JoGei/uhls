"""Target-dispatched floorplan hint emission helpers."""

from __future__ import annotations

from typing import Sequence

from .macros import MacroCollateral


def emit_floorplan_hints_tcl(
    target: str | None,
    *,
    design_name: str,
    top_module: str,
    macros: Sequence[MacroCollateral] = (),
) -> str:
    """Emit one target-appropriate floorplan-hints TCL template."""
    normalized = _normalize_target(target)
    if normalized == "ihp130":
        from .vendor.ihp130.floorplan import emit_ihp130_floorplan_hints_tcl

        return emit_ihp130_floorplan_hints_tcl(
            design_name=design_name,
            top_module=top_module,
            macros=macros,
        )
    return _emit_generic_floorplan_hints_tcl(
        design_name=design_name,
        top_module=top_module,
        macros=macros,
    )


def _emit_generic_floorplan_hints_tcl(
    *,
    design_name: str,
    top_module: str,
    macros: Sequence[MacroCollateral],
) -> str:
    lines = [
        f"# Floorplan hints for {design_name} ({top_module})",
        "#",
        "# This file is informational by default.",
        "# Copy the settings you want into ORFS config.mk as needed.",
        "# Example config.mk knobs:",
        "#   export CORE_UTILIZATION ?= 40",
        "#   export CORE_ASPECT_RATIO ?= 1.0",
        "#   export CORE_MARGIN ?= 4",
        "#   export MACRO_PLACE_HALO ?= 10 10",
        "#   export MACRO_PLACE_CHANNEL ?= 20 20",
        "#   export DIE_AREA ?= 0 0 500 500",
        "#   export CORE_AREA ?= 20 20 480 480",
    ]
    if macros:
        lines.append("")
        lines.append("# Macro instances present:")
        for macro in macros:
            lines.append(f"#   {macro.instance_name} : {macro.module_name}")
    lines.append("")
    return "\n".join(lines)


def _normalize_target(target: str | None) -> str:
    return "" if target is None else target.strip().lower()


__all__ = ["emit_floorplan_hints_tcl"]
