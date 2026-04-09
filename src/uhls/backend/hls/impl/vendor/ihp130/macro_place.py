"""IHP130 macro placement TCL helpers."""

from __future__ import annotations

from typing import Sequence

from uhls.backend.hls.impl.macros import MacroCollateral


def emit_ihp130_macro_placement_tcl(
    *,
    design_name: str,
    top_module: str,
    macros: Sequence[MacroCollateral] = (),
) -> str:
    """Emit one SG13G2-oriented macro-placement TCL template."""
    lines = [
        f"# IHP SG13G2 macro placement template for {design_name} ({top_module})",
        "#",
        "# ORFS can source this file through MACRO_PLACEMENT_TCL.",
        "# Replace the placeholder coordinates below with real placements.",
        "# Syntax:",
        "#   place_macro <instance_name> <x> <y> <orientation>",
        "# Common orientations: R0 R90 R180 R270 MX MY MXR90 MYR90",
    ]
    if macros:
        lines.append("")
        lines.append("# Macro instances discovered in this design:")
        for macro in macros:
            lines.append(
                f"# place_macro {macro.instance_name} <x> <y> R0  ;# {macro.module_name}"
            )
    else:
        lines.append("")
        lines.append("# No macros were discovered in this design.")
    lines.append("")
    return "\n".join(lines)


__all__ = ["emit_ihp130_macro_placement_tcl"]
