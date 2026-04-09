"""IHP130 PDN TCL helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

from uhls.backend.hls.impl.macros import MacroCollateral


def emit_ihp130_pdn_tcl(
    *,
    design_name: str,
    top_module: str,
    macros: Sequence[MacroCollateral] = (),
) -> str:
    """Emit one SG13G2-oriented PDN TCL file with SRAM macro hookups."""
    lines = [
        f"# IHP SG13G2 PDN TCL for {design_name} ({top_module})",
        "#",
        "# This is based on the public SG13G2 platform PDN setup, with",
        "# additional global connections for SRAM macro power pins.",
        "####################################",
        "# global connections",
        "####################################",
        "# standard cells",
        "add_global_connection -net {VDD} -pin_pattern {^VDD$} -power",
        "add_global_connection -net {VDD} -pin_pattern {^VDDPE$}",
        "add_global_connection -net {VDD} -pin_pattern {^VDDCE$}",
        "add_global_connection -net {VSS} -pin_pattern {^VSS$} -ground",
        "add_global_connection -net {VSS} -pin_pattern {^VSSE$}",
    ]
    if macros:
        lines.extend(
            [
                "# imported SRAM macros",
                "add_global_connection -net {VDD} -pin_pattern {^VDD!$} -power",
                "add_global_connection -net {VDD} -pin_pattern {^VDDARRAY!$} -power",
                "add_global_connection -net {VSS} -pin_pattern {^VSS!$} -ground",
            ]
        )
    lines.extend(
        [
            "global_connect",
            "####################################",
            "# voltage domains",
            "####################################",
            "set_voltage_domain -name {CORE} -power {VDD} -ground {VSS}",
            "#####################################",
            "# standard cell grid",
            "####################################",
            "define_pdn_grid -name {grid} -voltage_domains {CORE} -pins {TopMetal1 TopMetal2}",
            "add_pdn_ring -grid {grid} -layers {TopMetal1 TopMetal2} -widths {5.0} -spacings {2.0} \\",
            "  -core_offsets {4.5} -connect_to_pads",
            "add_pdn_stripe -grid {grid} -layer {Metal1} -width {0.44} -pitch {7.56} -offset {0} -followpins \\",
            "  -extend_to_core_ring",
            "add_pdn_stripe -grid {grid} -layer {TopMetal1} -width {2.200} -pitch {75.6} -offset {13.600} \\",
            "  -extend_to_core_ring",
            "add_pdn_stripe -grid {grid} -layer {TopMetal2} -width {2.200} -pitch {75.6} -offset {13.600} \\",
            "  -extend_to_core_ring",
            "add_pdn_connect -grid {grid} -layers {Metal1 TopMetal1}",
            "add_pdn_connect -grid {grid} -layers {TopMetal1 TopMetal2}",
        ]
    )
    macro_cells = _collect_physical_macro_cells(macros)
    if macro_cells:
        joined_cells = " ".join(macro_cells)
        lines.extend(
            [
                "#####################################",
                "# macro grid",
                "####################################",
                "define_pdn_grid \\",
                "  -name {CORE_macro_grid_1} -voltage_domains {CORE} \\",
                f"  -macro -cells {{{joined_cells}}} -grid_over_boundary",
                "add_pdn_connect -grid {CORE_macro_grid_1} -layers {Metal4 TopMetal1}",
                "add_pdn_connect -grid {CORE_macro_grid_1} -layers {Metal5 TopMetal1}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _collect_physical_macro_cells(macros: Sequence[MacroCollateral]) -> tuple[str, ...]:
    seen: set[str] = set()
    cells: list[str] = []
    for module_name in _iter_physical_macro_modules(macros):
        if module_name in seen:
            continue
        seen.add(module_name)
        cells.append(module_name)
    return tuple(cells)


def _iter_physical_macro_modules(macros: Sequence[MacroCollateral]) -> Iterable[str]:
    for macro in macros:
        if not macro.lef_files:
            continue
        if not macro.module_name:
            continue
        yield macro.module_name


__all__ = ["emit_ihp130_pdn_tcl"]
