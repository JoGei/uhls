"""Target-dispatched OpenROAD-flow-scripts config emission helpers."""

from __future__ import annotations

from typing import Sequence

from .macros import MacroCollateral


def emit_orfs_config(
    target: str | None,
    *,
    design_name: str,
    top_module: str,
    rtl_files: Sequence[str],
    sdc_file: str,
    macro_placement_tcl: str | None = None,
    pdn_tcl: str | None = None,
    macros: Sequence[MacroCollateral] = (),
) -> str:
    """Emit one target-appropriate ORFS config skeleton."""
    normalized = _normalize_target(target)
    if normalized == "ihp130":
        from .vendor.ihp130.orfs import emit_ihp130_orfs_config

        return emit_ihp130_orfs_config(
            design_name=design_name,
            top_module=top_module,
            rtl_files=rtl_files,
            sdc_file=sdc_file,
            macro_placement_tcl=macro_placement_tcl,
            pdn_tcl=pdn_tcl,
            macros=macros,
        )
    return _emit_generic_orfs_config(
        design_name=design_name,
        top_module=top_module,
        rtl_files=rtl_files,
        sdc_file=sdc_file,
        macro_placement_tcl=macro_placement_tcl,
        pdn_tcl=pdn_tcl,
        macros=macros,
    )


def emit_orfs_run_script(target: str | None, *, design_config: str = "config.mk") -> str:
    """Emit one target-appropriate ORFS launcher script."""
    normalized = _normalize_target(target)
    if normalized == "ihp130":
        from .vendor.ihp130.orfs import emit_ihp130_orfs_run_script

        return emit_ihp130_orfs_run_script(design_config=design_config)
    return _emit_generic_orfs_run_script(design_config=design_config)


def _emit_generic_orfs_config(
    *,
    design_name: str,
    top_module: str,
    rtl_files: Sequence[str],
    sdc_file: str,
    macro_placement_tcl: str | None,
    pdn_tcl: str | None,
    macros: Sequence[MacroCollateral],
) -> str:
    synthesis_macros = _collect_synthesis_macros(macros)
    verilog_files = _collect_synth_verilog_files(rtl_files, macros)
    include_dirs = _collect_include_dirs(synthesis_macros)
    defines = _collect_define_flags(synthesis_macros)
    synth_blackboxes = _collect_blackbox_modules(macros)
    lines = [
        f"export DESIGN_NAME = {design_name}",
        f"export PLATFORM = <set-platform>",
        f"export VERILOG_FILES = {' '.join(verilog_files) if verilog_files else '<set-rtl-files>'}",
        *([f"export VERILOG_INCLUDE_DIRS = {' '.join(include_dirs)}"] if include_dirs else []),
        *([f"export VERILOG_DEFINES = {' '.join(defines)}"] if defines else []),
        *([f"export SYNTH_BLACKBOXES = {' '.join(synth_blackboxes)}"] if synth_blackboxes else []),
        f"export SDC_FILE = {sdc_file}",
        f"export TOP_MODULE = {top_module}",
        *([f"export MACRO_PLACEMENT_TCL = {macro_placement_tcl}"] if macro_placement_tcl else []),
        *([f"export PDN_TCL = {pdn_tcl}"] if pdn_tcl else []),
        "export CORE_UTILIZATION ?= 40",
        "export CORE_ASPECT_RATIO ?= 1.0",
        "export CORE_MARGIN ?= 4",
        "export MACRO_PLACE_HALO ?= 10 10",
        "export MACRO_PLACE_CHANNEL ?= 20 20",
    ]
    if macros:
        lines.append(f"# macro instances: {', '.join(macro.instance_name for macro in macros)}")
    lines.append("")
    return "\n".join(lines)


def _normalize_target(target: str | None) -> str:
    return "" if target is None else target.strip().lower()


def _collect_synthesis_macros(macros: Sequence[MacroCollateral]) -> tuple[MacroCollateral, ...]:
    return tuple(macro for macro in macros if not _macro_requires_blackbox(macro))


def _macro_requires_blackbox(macro: MacroCollateral) -> bool:
    return bool(macro.liberty_files)


def _collect_blackbox_modules(macros: Sequence[MacroCollateral]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for macro in macros:
        if not _macro_requires_blackbox(macro):
            continue
        module_name = macro.module_name.strip()
        if module_name and module_name not in seen:
            seen.add(module_name)
            ordered.append(module_name)
    return ordered


def _collect_synth_verilog_files(rtl_files: Sequence[str], macros: Sequence[MacroCollateral]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for path in rtl_files:
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)
    for macro in _collect_synthesis_macros(macros):
        for path in macro.verilog_sources:
            if path and path not in seen:
                seen.add(path)
                ordered.append(path)
    return ordered


def _collect_include_dirs(macros: Sequence[MacroCollateral]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for macro in macros:
        for path in macro.include_dirs:
            if path and path not in seen:
                seen.add(path)
                ordered.append(path)
    return ordered


def _collect_define_flags(macros: Sequence[MacroCollateral]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for macro in macros:
        for define in macro.defines:
            define = define.strip()
            if not define:
                continue
            flag = define if define.startswith("-D") else f"-D{define}"
            if flag not in seen:
                seen.add(flag)
                ordered.append(flag)
    return ordered


def _emit_generic_orfs_run_script(*, design_config: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"',
            'ORFS_ROOT="${ORFS_ROOT:-${OPENROAD_FLOW_SCRIPTS_ROOT:-}}"',
            'if [[ -z "${ORFS_ROOT}" ]]; then',
            '  echo "error: define ORFS_ROOT or OPENROAD_FLOW_SCRIPTS_ROOT to your OpenROAD-flow-scripts checkout" >&2',
            "  exit 1",
            "fi",
            'if [[ -z "${YOSYS_EXE:-}" ]]; then',
            '  YOSYS_EXE="$(command -v yosys || true)"',
            '  if [[ -n "${YOSYS_EXE}" ]]; then export YOSYS_EXE; fi',
            "fi",
            'if [[ -z "${OPENROAD_EXE:-}" ]]; then',
            '  OPENROAD_EXE="$(command -v openroad || true)"',
            '  if [[ -n "${OPENROAD_EXE}" ]]; then export OPENROAD_EXE; fi',
            "fi",
            f'exec make --file="${{ORFS_ROOT}}/flow/Makefile" DESIGN_CONFIG="${{SCRIPT_DIR}}/{design_config}" "$@"',
            "",
        ]
    )


__all__ = ["emit_orfs_config", "emit_orfs_run_script"]
