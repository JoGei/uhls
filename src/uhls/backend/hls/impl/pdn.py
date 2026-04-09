"""Target-dispatched PDN TCL emission helpers."""

from __future__ import annotations

from typing import Sequence

from .macros import MacroCollateral


def emit_pdn_tcl(
    target: str | None,
    *,
    design_name: str,
    top_module: str,
    macros: Sequence[MacroCollateral] = (),
) -> str:
    """Emit one target-appropriate PDN TCL file."""
    normalized = _normalize_target(target)
    if normalized == "ihp130":
        from .vendor.ihp130.pdn import emit_ihp130_pdn_tcl

        return emit_ihp130_pdn_tcl(
            design_name=design_name,
            top_module=top_module,
            macros=macros,
        )
    return _emit_generic_pdn_tcl(
        design_name=design_name,
        top_module=top_module,
        macros=macros,
    )


def _emit_generic_pdn_tcl(
    *,
    design_name: str,
    top_module: str,
    macros: Sequence[MacroCollateral],
) -> str:
    lines = [
        f"# Generic PDN TCL for {design_name} ({top_module})",
        "# Replace this file with target-specific PDN setup as needed.",
    ]
    if macros:
        lines.append("# Macro instances present:")
        for macro in macros:
            lines.append(f"#   {macro.instance_name} : {macro.module_name}")
    lines.append("")
    return "\n".join(lines)


def _normalize_target(target: str | None) -> str:
    return "" if target is None else target.strip().lower()


__all__ = ["emit_pdn_tcl"]
