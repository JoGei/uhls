"""Fast ASIC area-estimate helpers built on ORFS synthesis reports."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AreaCellStat:
    """One cell-type area entry from the synthesis report."""

    cell_type: str
    count: int
    area_um2: float


@dataclass(frozen=True)
class AreaEstimateReport:
    """One parsed synth-only area estimate."""

    target: str
    module_name: str
    report_path: Path
    total_cells: int
    total_area_um2: float
    sequential_area_um2: float | None
    sequential_percent: float | None
    macro_cells: int
    macro_area_um2: float
    stdcell_cells: int
    stdcell_area_um2: float
    estimated_core_area_um2: float | None
    utilization_percent: float | None
    num_wires: int | None
    num_wire_bits: int | None
    num_public_wires: int | None
    num_public_wire_bits: int | None
    num_ports: int | None
    num_port_bits: int | None
    cell_stats: tuple[AreaCellStat, ...]


def estimate_area_from_orfs_bundle(target: str, bundle_dir: Path) -> AreaEstimateReport:
    """Run ORFS synth-only reporting for one emitted ASIC bundle and parse the result."""
    orfs_dir = bundle_dir / "orfs"
    run_script = orfs_dir / "run_orfs.sh"
    if not run_script.exists():
        raise ValueError(f"ASIC bundle '{bundle_dir}' is missing '{run_script.relative_to(bundle_dir)}'")
    env = os.environ.copy()
    completed = subprocess.run(
        [str(run_script), "synth-report"],
        cwd=orfs_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            tail = "\n".join(detail.splitlines()[-20:])
            raise RuntimeError(f"ORFS synth-report failed:\n{tail}")
        raise RuntimeError("ORFS synth-report failed")
    config_path = orfs_dir / "config.mk"
    design_name = _read_export_value(config_path, "DESIGN_NAME")
    platform_name = _read_export_value(config_path, "PLATFORM")
    if design_name is None or platform_name is None:
        raise RuntimeError(f"failed to resolve DESIGN_NAME/PLATFORM from '{config_path}'")
    report_path = orfs_dir / "reports" / platform_name / design_name / "base" / "synth_stat.txt"
    if not report_path.exists():
        raise RuntimeError(f"ORFS synth-report did not produce '{report_path}'")
    macros_path = orfs_dir / "macros.json"
    macro_modules = _load_macro_modules(macros_path)
    utilization_percent = _read_float_export_value(config_path, "CORE_UTILIZATION")
    return parse_yosys_synth_stat(
        report_path.read_text(encoding="utf-8"),
        target=target,
        module_name=design_name,
        report_path=report_path,
        macro_modules=macro_modules,
        utilization_percent=utilization_percent,
    )


def parse_yosys_synth_stat(
    text: str,
    *,
    target: str,
    module_name: str,
    report_path: Path,
    macro_modules: set[str] | None = None,
    utilization_percent: float | None = None,
) -> AreaEstimateReport:
    """Parse one ORFS/Yosys synth_stat.txt file into one structured report."""
    macro_modules = set() if macro_modules is None else {name.strip() for name in macro_modules if name.strip()}
    section_lines = _extract_module_section(text, module_name)
    parsed_module_name = module_name
    total_cells: int | None = None
    coarse_area: float | None = None
    sequential_area_um2: float | None = None
    sequential_percent: float | None = None
    num_wires: int | None = None
    num_wire_bits: int | None = None
    num_public_wires: int | None = None
    num_public_wire_bits: int | None = None
    num_ports: int | None = None
    num_port_bits: int | None = None
    cell_stats: list[AreaCellStat] = []

    in_cell_table = False
    for raw_line in section_lines:
        line = raw_line.rstrip()
        if not line.strip():
            if in_cell_table:
                in_cell_table = False
            continue
        if match := re.match(r"^===\s+(.+?)\s+===$", line):
            parsed_module_name = _normalize_module_name(match.group(1))
            continue
        if match := re.match(r"^\s*(\d+)\s+- wires$", line):
            num_wires = int(match.group(1))
            continue
        if match := re.match(r"^\s*(\d+)\s+- wire bits$", line):
            num_wire_bits = int(match.group(1))
            continue
        if match := re.match(r"^\s*(\d+)\s+- public wires$", line):
            num_public_wires = int(match.group(1))
            continue
        if match := re.match(r"^\s*(\d+)\s+- public wire bits$", line):
            num_public_wire_bits = int(match.group(1))
            continue
        if match := re.match(r"^\s*(\d+)\s+- ports$", line):
            num_ports = int(match.group(1))
            continue
        if match := re.match(r"^\s*(\d+)\s+- port bits$", line):
            num_port_bits = int(match.group(1))
            continue
        if match := re.match(r"^\s*(\d+)\s+([0-9.eE+-]+)\s+cells$", line):
            total_cells = int(match.group(1))
            coarse_area = float(match.group(2))
            in_cell_table = True
            continue
        if in_cell_table and (match := re.match(r"^\s*(\d+)\s+([0-9.eE+-]+)\s+(\S+)\s*$", line)):
            cell_stats.append(
                AreaCellStat(
                    cell_type=match.group(3),
                    count=int(match.group(1)),
                    area_um2=float(match.group(2)),
                )
            )
            continue
        if match := re.match(r"^\s*Chip area for module '([^']+)':\s+([0-9.eE+-]+)\s*$", line):
            parsed_module_name = _normalize_module_name(match.group(1))
            coarse_area = float(match.group(2))
            continue
        if match := re.match(
            r"^\s*of which used for sequential elements:\s+([0-9.eE+-]+)\s+\(([0-9.]+)%\)\s*$",
            line,
        ):
            sequential_area_um2 = float(match.group(1))
            sequential_percent = float(match.group(2))
            continue

    if total_cells is None:
        raise ValueError(f"failed to parse total cell count for module '{module_name}' from '{report_path}'")
    if coarse_area is None:
        raise ValueError(f"failed to parse total area for module '{module_name}' from '{report_path}'")

    macro_area_um2 = sum(entry.area_um2 for entry in cell_stats if entry.cell_type in macro_modules)
    macro_cells = sum(entry.count for entry in cell_stats if entry.cell_type in macro_modules)
    stdcell_area_um2 = coarse_area - macro_area_um2
    stdcell_cells = total_cells - macro_cells
    estimated_core_area_um2 = None
    if utilization_percent is not None and utilization_percent > 0:
        estimated_core_area_um2 = coarse_area / (utilization_percent / 100.0)
    return AreaEstimateReport(
        target=target,
        module_name=parsed_module_name,
        report_path=report_path,
        total_cells=total_cells,
        total_area_um2=coarse_area,
        sequential_area_um2=sequential_area_um2,
        sequential_percent=sequential_percent,
        macro_cells=macro_cells,
        macro_area_um2=macro_area_um2,
        stdcell_cells=stdcell_cells,
        stdcell_area_um2=stdcell_area_um2,
        estimated_core_area_um2=estimated_core_area_um2,
        utilization_percent=utilization_percent,
        num_wires=num_wires,
        num_wire_bits=num_wire_bits,
        num_public_wires=num_public_wires,
        num_public_wire_bits=num_public_wire_bits,
        num_ports=num_ports,
        num_port_bits=num_port_bits,
        cell_stats=tuple(sorted(cell_stats, key=lambda entry: (-entry.area_um2, entry.cell_type))),
    )


def _extract_module_section(text: str, module_name: str) -> list[str]:
    lines = text.splitlines()
    wanted = _normalize_module_name(module_name)
    current_name: str | None = None
    collecting = False
    section: list[str] = []
    for line in lines:
        if match := re.match(r"^===\s+(.+?)\s+===$", line):
            current_name = _normalize_module_name(match.group(1))
            collecting = current_name == wanted
            if collecting:
                section = [line]
            elif section:
                break
            continue
        if collecting:
            section.append(line)
    if not section:
        raise ValueError(f"failed to find module section '{module_name}' in synth_stat")
    return section


def _normalize_module_name(name: str) -> str:
    stripped = name.strip()
    return stripped[1:] if stripped.startswith("\\") else stripped


def _read_export_value(config_path: Path, name: str) -> str | None:
    if not config_path.exists():
        return None
    pattern = re.compile(rf"^export\s+{re.escape(name)}\s*(?:\?|)?=\s*(.+?)\s*$")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is not None:
            return match.group(1).strip()
    return None


def _read_float_export_value(config_path: Path, name: str) -> float | None:
    value = _read_export_value(config_path, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_macro_modules(macros_path: Path) -> set[str]:
    if not macros_path.exists():
        return set()
    payload = json.loads(macros_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return set()
    modules: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if not any(isinstance(entry.get(key), list) and entry.get(key) for key in ("lef_files", "liberty_files", "gds_files")):
            continue
        module_name = entry.get("module")
        if isinstance(module_name, str) and module_name.strip():
            modules.add(module_name.strip())
    return modules


__all__ = [
    "AreaCellStat",
    "AreaEstimateReport",
    "estimate_area_from_orfs_bundle",
    "parse_yosys_synth_stat",
]
