"""Implementation collateral for concrete backend targets and vendor flows."""

from .analytic import AnalyticalAreaItem, AnalyticalAreaReport, estimate_analytical_area
from .area import AreaCellStat, AreaEstimateReport, estimate_area_from_orfs_bundle, parse_yosys_synth_stat
from .macros import MacroCollateral, collect_flow_macros
from .floorplan import emit_floorplan_hints_tcl
from .macro_place import emit_macro_placement_tcl
from .orfs import emit_orfs_config, emit_orfs_run_script
from .pdn import emit_pdn_tcl
from .select import MEM_POLICIES, MemoryPolicy, parse_memory_policy, select_memory_implementation
from .sdc import emit_sdc

__all__ = [
    "AnalyticalAreaItem",
    "AnalyticalAreaReport",
    "AreaCellStat",
    "AreaEstimateReport",
    "MEM_POLICIES",
    "MacroCollateral",
    "MemoryPolicy",
    "collect_flow_macros",
    "estimate_analytical_area",
    "estimate_area_from_orfs_bundle",
    "emit_floorplan_hints_tcl",
    "emit_macro_placement_tcl",
    "emit_orfs_config",
    "emit_orfs_run_script",
    "emit_pdn_tcl",
    "emit_sdc",
    "parse_yosys_synth_stat",
    "parse_memory_policy",
    "select_memory_implementation",
]
