"""IHP130 floorplan hint helpers."""

from __future__ import annotations

from typing import Sequence

from uhls.backend.hls.impl.macros import MacroCollateral


def emit_ihp130_floorplan_hints_tcl(
    *,
    design_name: str,
    top_module: str,
    macros: Sequence[MacroCollateral] = (),
) -> str:
    """Emit one SG13G2-oriented floorplan hint template."""
    lines = [
        f"# IHP SG13G2 floorplan hints for {design_name} ({top_module})",
        "#",
        "# Suggested starting points for ORFS config.mk:",
        "#   export CORE_UTILIZATION ?= 35",
        "#   export CORE_ASPECT_RATIO ?= 1.0",
        "#   export CORE_MARGIN ?= 4",
        "#   export MACRO_PLACE_HALO ?= 10 10",
        "#   export MACRO_PLACE_CHANNEL ?= 20 20",
        "#",
        "# If macro placement becomes constrained, uncomment and tune area hints:",
        "#   export DIE_AREA ?= 0 0 600 600",
        "#   export CORE_AREA ?= 40 40 560 560",
    ]
    if macros:
        lines.append("")
        lines.append("# Macro instances present:")
        for macro in macros:
            lines.append(f"#   {macro.instance_name} : {macro.module_name}")
    else:
        lines.append("")
        lines.append("# No macros were discovered in this design.")
    lines.append("")
    return "\n".join(lines)


__all__ = ["emit_ihp130_floorplan_hints_tcl"]
