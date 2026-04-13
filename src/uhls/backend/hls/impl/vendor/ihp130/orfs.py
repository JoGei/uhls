"""IHP130 OpenROAD-flow-scripts config helpers."""

from __future__ import annotations

from typing import Sequence

from uhls.backend.hls.impl.macros import MacroCollateral
from uhls.backend.hls.impl.orfs import (
    _collect_blackbox_modules,
    _collect_define_flags,
    _collect_include_dirs,
    _collect_synthesis_macros,
    _collect_synth_verilog_files,
)


def emit_ihp130_orfs_config(
    *,
    design_name: str,
    top_module: str,
    rtl_files: Sequence[str],
    sdc_file: str,
    macro_placement_tcl: str | None = None,
    pdn_tcl: str | None = None,
    macros: Sequence[MacroCollateral] = (),
) -> str:
    """Emit a small ORFS config skeleton for IHP SG13G2."""
    synthesis_macros = _collect_synthesis_macros(macros)
    verilog_files = _collect_synth_verilog_files(rtl_files, macros)
    include_dirs = _collect_include_dirs(synthesis_macros)
    defines = _collect_define_flags(synthesis_macros)
    synth_blackboxes = _collect_blackbox_modules(macros)
    macro_lines = _emit_macro_lines(macros)
    define_lines = [f"export VERILOG_DEFINES = {' '.join(defines)}"] if defines else []
    return "\n".join(
        [
            f"export DESIGN_NAME = {design_name}",
            "export PLATFORM = ihp-sg13g2",
            f"export VERILOG_FILES = {' '.join(verilog_files) if verilog_files else '<set-rtl-files>'}",
            *([f"export VERILOG_INCLUDE_DIRS = {' '.join(include_dirs)}"] if include_dirs else []),
            *([f"export SYNTH_BLACKBOXES = {' '.join(synth_blackboxes)}"] if synth_blackboxes else []),
            f"export SDC_FILE = {sdc_file}",
            f"export TOP_MODULE = {top_module}",
            *([f"export MACRO_PLACEMENT_TCL = {macro_placement_tcl}"] if macro_placement_tcl else []),
            *([f"export PDN_TCL = {pdn_tcl}"] if pdn_tcl else []),
            "export CORE_UTILIZATION ?= 35",
            "export CORE_ASPECT_RATIO ?= 1.0",
            "export HOLD_SLACK_MARGIN ?= 0.03",
            "export MACRO_PLACE_HALO ?= 10 10",
            "export MACRO_PLACE_CHANNEL ?= 20 20",
            *define_lines,
            *macro_lines,
            "",
            "# Optional: set DIE_AREA / CORE_AREA explicitly as the design matures.",
            "# See floorplan_hints.tcl for a small template of starting values.",
        ]
    )


def emit_ihp130_orfs_run_script(*, design_config: str = "config.mk") -> str:
    """Emit one SG13G2-oriented ORFS launcher script."""
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
            '  if [[ -x "${ORFS_ROOT}/tools/install/OpenROAD/bin/openroad" ]]; then',
            '    OPENROAD_EXE="${ORFS_ROOT}/tools/install/OpenROAD/bin/openroad"',
            '    export OPENROAD_EXE',
            "  fi",
            "fi",
            'if [[ -z "${OPENROAD_EXE:-}" ]]; then',
            '  OPENROAD_EXE="$(command -v openroad || true)"',
            '  if [[ -n "${OPENROAD_EXE}" ]]; then export OPENROAD_EXE; fi',
            "fi",
            'FLOW_SHIM_ROOT="${SCRIPT_DIR}/.orfs_flow_shim"',
            'mkdir -p "${FLOW_SHIM_ROOT}/scripts"',
            'for subdir in designs platforms util test; do',
            '  ln -sfn "${ORFS_ROOT}/flow/${subdir}" "${FLOW_SHIM_ROOT}/${subdir}"',
            "done",
            'for path in "${ORFS_ROOT}/flow/scripts"/*; do',
            '  name="$(basename "${path}")"',
            '  if [[ "${name}" == "flow.sh" ]]; then',
            "    continue",
            "  fi",
            '  ln -sfn "${path}" "${FLOW_SHIM_ROOT}/scripts/${name}"',
            "done",
            'cat > "${FLOW_SHIM_ROOT}/scripts/flow.sh" <<\'EOF\'',
            '#!/usr/bin/env bash',
            'set -euo pipefail',
            '',
            'mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$REPORTS_DIR" "$OBJECTS_DIR"',
            '',
            'echo "Running $2.tcl, stage $1"',
            '',
            '(',
            '  trap \'if [[ -f "$LOG_DIR/$1.tmp.log" ]]; then mv "$LOG_DIR/$1.tmp.log" "$LOG_DIR/$1.log"; fi\' EXIT',
            '',
            '  $OPENROAD_CMD -no_splash "$SCRIPTS_DIR/$2.tcl" -metrics "$LOG_DIR/$1.json" \\',
            '    2>&1 | tee "$LOG_DIR/$1.tmp.log"',
            ')',
            '',
            '"$PYTHON_EXE" "$UTILS_DIR/genElapsedTime.py" --match "$1" -d "$LOG_DIR" \\',
            '  | tee -a "$(realpath "$LOG_DIR/$1.log")"',
            'EOF',
            'chmod +x "${FLOW_SHIM_ROOT}/scripts/flow.sh"',
            f'exec make --file="${{ORFS_ROOT}}/flow/Makefile" FLOW_HOME="${{FLOW_SHIM_ROOT}}" DESIGN_CONFIG="${{SCRIPT_DIR}}/{design_config}" "$@"',
            "",
        ]
    )


def _emit_macro_lines(macros: Sequence[MacroCollateral]) -> list[str]:
    if not macros:
        return []
    lef_files = sorted({path for macro in macros for path in macro.lef_files})
    liberty_files = sorted({path for macro in macros for path in macro.liberty_files})
    gds_files = sorted({path for macro in macros for path in macro.gds_files})
    slow_libs, fast_libs, typ_libs, uncategorized_libs = _classify_corner_liberty_files(liberty_files)
    lines = [f"# macro instances: {', '.join(macro.instance_name for macro in macros)}"]
    if lef_files:
        lines.append(f"export ADDITIONAL_LEFS = {' '.join(lef_files)}")
    if liberty_files:
        lines.append(f"export ADDITIONAL_LIBS = {' '.join(liberty_files)}")
    if slow_libs:
        lines.append(f"export ADDITIONAL_SLOW_LIBS = {' '.join(slow_libs)}")
    if fast_libs:
        lines.append(f"export ADDITIONAL_FAST_LIBS = {' '.join(fast_libs)}")
    if typ_libs or uncategorized_libs:
        lines.append(f"export ADDITIONAL_TYP_LIBS = {' '.join((*typ_libs, *uncategorized_libs))}")
    if gds_files:
        lines.append(f"export ADDITIONAL_GDS = {' '.join(gds_files)}")
    return lines


def _classify_corner_liberty_files(liberty_files: Sequence[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    slow: list[str] = []
    fast: list[str] = []
    typ: list[str] = []
    uncategorized: list[str] = []
    for path in liberty_files:
        lower = path.lower()
        if "_slow_" in lower:
            slow.append(path)
        elif "_fast_" in lower:
            fast.append(path)
        elif "_typ_" in lower:
            typ.append(path)
        else:
            uncategorized.append(path)
    return slow, fast, typ, uncategorized


__all__ = ["emit_ihp130_orfs_config", "emit_ihp130_orfs_run_script"]
