"""Target-dispatched macro placement template emission helpers."""

from __future__ import annotations

from typing import Sequence

from .macros import MacroCollateral


def emit_macro_placement_tcl(
    target: str | None,
    *,
    design_name: str,
    top_module: str,
    macros: Sequence[MacroCollateral] = (),
) -> str:
    """Emit one target-appropriate macro-placement TCL template."""
    normalized = _normalize_target(target)
    if normalized == "ihp130":
        from .vendor.ihp130.macro_place import emit_ihp130_macro_placement_tcl

        return emit_ihp130_macro_placement_tcl(
            design_name=design_name,
            top_module=top_module,
            macros=macros,
        )
    return _emit_generic_macro_placement_tcl(
        design_name=design_name,
        top_module=top_module,
        macros=macros,
    )


def _emit_generic_macro_placement_tcl(
    *,
    design_name: str,
    top_module: str,
    macros: Sequence[MacroCollateral],
) -> str:
    lines = [
        f"# Macro placement template for {design_name} ({top_module})",
        "#",
        "# Fill in coordinates/orientations as needed.",
        "# Example:",
        "# place_macro <instance_name> <x> <y> <orientation>",
    ]
    if macros:
        lines.append("")
        lines.append("# Macro instances discovered in this design:")
        for macro in macros:
            lines.append(f"# place_macro {macro.instance_name} <x> <y> R0")
    else:
        lines.append("")
        lines.append("# No macros were discovered in this design.")
    lines.append("")
    return "\n".join(lines)


def _normalize_target(target: str | None) -> str:
    return "" if target is None else target.strip().lower()


__all__ = ["emit_macro_placement_tcl"]
