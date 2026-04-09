from __future__ import annotations

from pathlib import Path
import unittest

from uhls.backend.hls.impl import (
    estimate_analytical_area,
    collect_flow_macros,
    emit_floorplan_hints_tcl,
    emit_macro_placement_tcl,
    emit_orfs_config,
    emit_orfs_run_script,
    emit_pdn_tcl,
    emit_sdc,
    parse_yosys_synth_stat,
)
from uhls.backend.hls.lib import validate_component_library
from uhls.backend.hls.uglir import UGLIRDesign, UGLIRResource
from uhls.backend.hls.uglir import UGLIRMux, UGLIRMuxCase


class ImplTests(unittest.TestCase):
    def test_estimate_analytical_area_uses_component_library_ppa_metadata(self) -> None:
        report = estimate_analytical_area(
            UGLIRDesign(
                name="wrapped",
                resources=[
                    UGLIRResource("inst", "alu0", "GEN_ALU", "GEN_ALU"),
                    UGLIRResource("inst", "alu1", "GEN_ALU", "GEN_ALU"),
                    UGLIRResource("reg", "r0_q", "i32"),
                ],
                muxes=[
                    UGLIRMux(
                        "mux0",
                        "i32",
                        "sel",
                        [UGLIRMuxCase("false", "a"), UGLIRMuxCase("true", "b")],
                    )
                ],
            ),
            component_library=validate_component_library(
                {
                    "GEN_ALU": {
                        "kind": "combinational",
                        "ppa_estimate": {
                            "area_um2": 42,
                            "power_mW": 0.01,
                            "performance_Hz": 300000000,
                        },
                        "ports": {
                            "a": {"dir": "input", "type": "i32"},
                            "y": {"dir": "output", "type": "i32"},
                        },
                        "supports": {"add": {"ii": 1, "d": 1}},
                    }
                }
            ),
        )

        self.assertEqual(report.design_name, "wrapped")
        self.assertEqual(report.stage, "uglir")
        self.assertAlmostEqual(report.total_area_um2, 124.0)
        self.assertEqual(report.items[0].category, "component")
        self.assertEqual(report.items[0].label, "GEN_ALU")
        self.assertEqual(report.items[0].count, 2)
        self.assertEqual(report.items[0].source, "ppa_estimate")
        self.assertFalse(report.warnings)

    def test_parse_yosys_synth_stat(self) -> None:
        report = parse_yosys_synth_stat(
            """
=== dot4_relu ===

        +----------Local Count, excluding submodules.
        |        +-Local Area, excluding submodules.
        |        |
     5748        - wires
     5844        - wire bits
      450        - public wires
      546        - public wire bits
       11        - ports
      107        - port bits
     5709 1.69E+05 cells
        2  9.9E+04   RM_IHPSG13_1P_256x32_c2_bm_bist
      329 1.61E+04   sg13g2_dfrbpq_1
      264 3.83E+03   sg13g2_xor2_1

   Chip area for module '\\dot4_relu': 168563.812200
     of which used for sequential elements: 16117.315200 (9.56%)
""",
            target="ihp130",
            module_name="dot4_relu",
            report_path=Path("synth_stat.txt"),
            macro_modules={"RM_IHPSG13_1P_256x32_c2_bm_bist"},
            utilization_percent=35.0,
        )
        self.assertEqual(report.module_name, "dot4_relu")
        self.assertEqual(report.total_cells, 5709)
        self.assertAlmostEqual(report.total_area_um2, 168563.8122)
        self.assertAlmostEqual(report.sequential_area_um2 or 0.0, 16117.3152)
        self.assertAlmostEqual(report.sequential_percent or 0.0, 9.56)
        self.assertEqual(report.macro_cells, 2)
        self.assertAlmostEqual(report.macro_area_um2, 99000.0)
        self.assertEqual(report.stdcell_cells, 5707)
        self.assertAlmostEqual(report.stdcell_area_um2, 69563.8122)
        self.assertAlmostEqual(report.estimated_core_area_um2 or 0.0, 168563.8122 / 0.35)
        self.assertEqual(report.num_wires, 5748)
        self.assertEqual(report.num_ports, 11)
        self.assertEqual(report.cell_stats[0].cell_type, "RM_IHPSG13_1P_256x32_c2_bm_bist")

    def test_emit_generic_sdc(self) -> None:
        rendered = emit_sdc(None, clock_port="clk", clock_name="clk", clock_period_ns=5.0)
        self.assertIn("create_clock [get_ports {clk}] -name clk -period 5", rendered)

    def test_emit_ihp130_orfs_config(self) -> None:
        rendered = emit_orfs_config(
            "ihp130",
            design_name="dot4_relu_core",
            top_module="dot4_relu_core",
            rtl_files=("build/dot4_relu.v",),
            sdc_file="build/dot4_relu.sdc",
            macro_placement_tcl="build/macro_place.tcl",
            pdn_tcl="build/pdn.tcl",
            macros=(),
        )
        self.assertIn("export PLATFORM = ihp-sg13g2", rendered)
        self.assertIn("export DESIGN_NAME = dot4_relu_core", rendered)
        self.assertIn("export SDC_FILE = build/dot4_relu.sdc", rendered)
        self.assertIn("export MACRO_PLACEMENT_TCL = build/macro_place.tcl", rendered)
        self.assertIn("export PDN_TCL = build/pdn.tcl", rendered)
        self.assertIn("export CORE_UTILIZATION ?= 35", rendered)
        self.assertIn("export HOLD_SLACK_MARGIN ?= 0.03", rendered)
        self.assertIn("export MACRO_PLACE_HALO ?= 10 10", rendered)

    def test_emit_ihp130_orfs_config_includes_macro_collateral(self) -> None:
        rendered = emit_orfs_config(
            "ihp130",
            design_name="dot4_relu_core",
            top_module="dot4_relu_core",
            rtl_files=("build/dot4_relu.v",),
            sdc_file="build/dot4_relu.sdc",
            macro_placement_tcl="build/macro_place.tcl",
            pdn_tcl="build/pdn.tcl",
            macros=collect_flow_macros(
                "ihp130",
                UGLIRDesign(
                    name="wrapped",
                    resources=[
                        UGLIRResource(
                            "inst",
                            "A_mem_inst",
                            "RM_IHPSG13_1P_64x64_c2_bm_bist<W=64,DEPTH=64>",
                            "RM_IHPSG13_1P_64x64_c2_bm_bist<word_t=i64,word_len=64>",
                        ),
                        UGLIRResource("inst", "gen_alu0", "GEN_ALU", "GEN_ALU"),
                    ],
                ),
                {
                    "RM_IHPSG13_1P_64x64_c2_bm_bist": {
                        "kind": "memory",
                        "parameters": {
                            "word_t": {"kind": "type"},
                            "word_len": {"kind": "int"},
                        },
                        "hdl": {
                            "language": "verilog",
                            "module": "RM_IHPSG13_1P_64x64_c2_bm_bist",
                            "sources": ["${IHP130_PDK_ROOT}/mem.v", "${IHP130_PDK_ROOT}/core.v"],
                            "include_dirs": ["${IHP130_PDK_ROOT}/include"],
                            "defines": ["FUNCTIONAL"],
                            "lef_files": ["${IHP130_PDK_ROOT}/lef/mem.lef"],
                            "liberty_files": ["${IHP130_PDK_ROOT}/lib/mem_typ.lib"],
                            "gds_files": ["${IHP130_PDK_ROOT}/gds/mem.gds"],
                        },
                        "ports": {
                            "A_CLK": {"dir": "input", "type": "clock"},
                            "A_DOUT": {"dir": "output", "type": "i64"},
                        },
                        "supports": {"load": {"ii": 1, "d": 2}, "store": {"ii": 1, "d": 1}},
                        "memory": {"word_t": "i64", "word_len": 64},
                    },
                    "GEN_ALU": {
                        "kind": "combinational",
                        "hdl": {"language": "verilog", "module": "GEN_ALU", "source": "gen_alu.v"},
                        "ports": {"a": {"dir": "input", "type": "i64"}, "y": {"dir": "output", "type": "i64"}},
                        "supports": {"add": {"ii": 1, "d": 1}},
                    }
                },
            ),
        )
        self.assertIn("export VERILOG_FILES = build/dot4_relu.v gen_alu.v", rendered)
        self.assertIn("export SYNTH_BLACKBOXES = RM_IHPSG13_1P_64x64_c2_bm_bist", rendered)
        self.assertIn("export PDN_TCL = build/pdn.tcl", rendered)
        self.assertNotIn("${IHP130_PDK_ROOT}/mem.v", rendered)
        self.assertNotIn("${IHP130_PDK_ROOT}/core.v", rendered)
        self.assertNotIn("export VERILOG_INCLUDE_DIRS = ${IHP130_PDK_ROOT}/include", rendered)
        self.assertNotIn("export VERILOG_DEFINES = -DFUNCTIONAL", rendered)
        self.assertIn("export ADDITIONAL_LEFS = ${IHP130_PDK_ROOT}/lef/mem.lef", rendered)
        self.assertIn("export ADDITIONAL_LIBS = ${IHP130_PDK_ROOT}/lib/mem_typ.lib", rendered)
        self.assertIn("export ADDITIONAL_TYP_LIBS = ${IHP130_PDK_ROOT}/lib/mem_typ.lib", rendered)
        self.assertIn("export ADDITIONAL_GDS = ${IHP130_PDK_ROOT}/gds/mem.gds", rendered)

    def test_emit_ihp130_macro_placement_tcl_lists_macro_instances(self) -> None:
        rendered = emit_macro_placement_tcl(
            "ihp130",
            design_name="dot4_relu_core",
            top_module="dot4_relu_core",
            macros=(
                collect_flow_macros(
                    "ihp130",
                    UGLIRDesign(
                        name="wrapped",
                        resources=[
                            UGLIRResource(
                                "inst",
                                "A_mem_inst",
                                "RM_IHPSG13_1P_64x64_c2_bm_bist<W=64,DEPTH=64>",
                                "RM_IHPSG13_1P_64x64_c2_bm_bist<word_t=i64,word_len=64>",
                            )
                        ],
                    ),
                    {
                        "RM_IHPSG13_1P_64x64_c2_bm_bist": {
                            "kind": "memory",
                            "parameters": {
                                "word_t": {"kind": "type"},
                                "word_len": {"kind": "int"},
                            },
                            "hdl": {
                                "language": "verilog",
                                "module": "RM_IHPSG13_1P_64x64_c2_bm_bist",
                                "source": "${IHP130_PDK_ROOT}/mem.v",
                                "lef_files": ["${IHP130_PDK_ROOT}/lef/mem.lef"],
                            },
                            "ports": {
                                "A_CLK": {"dir": "input", "type": "clock"},
                                "A_DOUT": {"dir": "output", "type": "i64"},
                            },
                            "supports": {"load": {"ii": 1, "d": 2}, "store": {"ii": 1, "d": 1}},
                            "memory": {"word_t": "i64", "word_len": 64},
                        }
                    },
                )
            ),
        )
        self.assertIn("# ORFS can source this file through MACRO_PLACEMENT_TCL.", rendered)
        self.assertIn("# place_macro A_mem_inst <x> <y> R0", rendered)

    def test_emit_ihp130_pdn_tcl_adds_macro_power_connections(self) -> None:
        rendered = emit_pdn_tcl(
            "ihp130",
            design_name="dot4_relu_core",
            top_module="dot4_relu_core",
            macros=(
                collect_flow_macros(
                    "ihp130",
                    UGLIRDesign(
                        name="wrapped",
                        resources=[
                            UGLIRResource(
                                "inst",
                                "A_mem_inst",
                                "RM_IHPSG13_1P_64x64_c2_bm_bist<W=64,DEPTH=64>",
                                "RM_IHPSG13_1P_64x64_c2_bm_bist<word_t=i64,word_len=64>",
                            )
                        ],
                    ),
                    {
                        "RM_IHPSG13_1P_64x64_c2_bm_bist": {
                            "kind": "memory",
                            "parameters": {
                                "word_t": {"kind": "type"},
                                "word_len": {"kind": "int"},
                            },
                            "hdl": {
                                "language": "verilog",
                                "module": "RM_IHPSG13_1P_64x64_c2_bm_bist",
                                "source": "${IHP130_PDK_ROOT}/mem.v",
                                "lef_files": ["${IHP130_PDK_ROOT}/lef/mem.lef"],
                            },
                            "ports": {
                                "A_CLK": {"dir": "input", "type": "clock"},
                                "A_DOUT": {"dir": "output", "type": "i64"},
                            },
                            "supports": {"load": {"ii": 1, "d": 2}, "store": {"ii": 1, "d": 1}},
                            "memory": {"word_t": "i64", "word_len": 64},
                        }
                    },
                )
            ),
        )
        self.assertIn("add_global_connection -net {VDD} -pin_pattern {^VDD!$} -power", rendered)
        self.assertIn("add_global_connection -net {VDD} -pin_pattern {^VDDARRAY!$} -power", rendered)
        self.assertIn("add_global_connection -net {VSS} -pin_pattern {^VSS!$} -ground", rendered)
        self.assertIn("-macro -cells {RM_IHPSG13_1P_64x64_c2_bm_bist} -grid_over_boundary", rendered)
        self.assertIn("add_pdn_connect -grid {CORE_macro_grid_1} -layers {Metal4 TopMetal1}", rendered)

    def test_emit_ihp130_orfs_run_script_uses_design_config(self) -> None:
        rendered = emit_orfs_run_script("ihp130", design_config="config.mk")
        self.assertIn('ORFS_ROOT="${ORFS_ROOT:-${OPENROAD_FLOW_SCRIPTS_ROOT:-}}"', rendered)
        self.assertIn('YOSYS_EXE="$(command -v yosys || true)"', rendered)
        self.assertIn('OPENROAD_EXE="$(command -v openroad || true)"', rendered)
        self.assertIn('FLOW_SHIM_ROOT="${SCRIPT_DIR}/.orfs_flow_shim"', rendered)
        self.assertIn('if [[ -f "$LOG_DIR/$1.tmp.log" ]]; then mv "$LOG_DIR/$1.tmp.log" "$LOG_DIR/$1.log"; fi', rendered)
        self.assertIn('FLOW_HOME="${FLOW_SHIM_ROOT}"', rendered)
        self.assertIn('DESIGN_CONFIG="${SCRIPT_DIR}/config.mk"', rendered)

    def test_emit_ihp130_floorplan_hints_tcl_lists_defaults_and_macros(self) -> None:
        rendered = emit_floorplan_hints_tcl(
            "ihp130",
            design_name="dot4_relu_core",
            top_module="dot4_relu_core",
            macros=(
                collect_flow_macros(
                    "ihp130",
                    UGLIRDesign(
                        name="wrapped",
                        resources=[
                            UGLIRResource(
                                "inst",
                                "A_mem_inst",
                                "RM_IHPSG13_1P_64x64_c2_bm_bist<W=64,DEPTH=64>",
                                "RM_IHPSG13_1P_64x64_c2_bm_bist<word_t=i64,word_len=64>",
                            )
                        ],
                    ),
                    {
                        "RM_IHPSG13_1P_64x64_c2_bm_bist": {
                            "kind": "memory",
                            "parameters": {
                                "word_t": {"kind": "type"},
                                "word_len": {"kind": "int"},
                            },
                            "hdl": {
                                "language": "verilog",
                                "module": "RM_IHPSG13_1P_64x64_c2_bm_bist",
                                "source": "${IHP130_PDK_ROOT}/mem.v",
                            },
                            "ports": {
                                "A_CLK": {"dir": "input", "type": "clock"},
                                "A_DOUT": {"dir": "output", "type": "i64"},
                            },
                            "supports": {"load": {"ii": 1, "d": 2}, "store": {"ii": 1, "d": 1}},
                            "memory": {"word_t": "i64", "word_len": 64},
                        }
                    },
                )
            ),
        )
        self.assertIn("#   export CORE_UTILIZATION ?= 35", rendered)
        self.assertIn("#   export MACRO_PLACE_CHANNEL ?= 20 20", rendered)
        self.assertIn("#   A_mem_inst : RM_IHPSG13_1P_64x64_c2_bm_bist", rendered)

    def test_collect_ihp130_macros_from_instantiated_design(self) -> None:
        design = UGLIRDesign(
            name="wrapped",
            resources=[
                UGLIRResource(
                    "inst",
                    "A_mem_inst",
                    "RM_IHPSG13_1P_64x64_c2_bm_bist<W=64,DEPTH=64>",
                    "RM_IHPSG13_1P_64x64_c2_bm_bist<word_t=i64,word_len=64>",
                ),
                UGLIRResource("inst", "gen_alu0", "GEN_ALU", "GEN_ALU"),
            ],
        )
        component_library = {
            "RM_IHPSG13_1P_64x64_c2_bm_bist": {
                "kind": "memory",
                "parameters": {
                    "word_t": {"kind": "type"},
                    "word_len": {"kind": "int"},
                },
                "hdl": {
                    "language": "verilog",
                    "module": "RM_IHPSG13_1P_64x64_c2_bm_bist",
                    "sources": ["${IHP130_PDK_ROOT}/mem.v", "${IHP130_PDK_ROOT}/core.v"],
                    "defines": ["FUNCTIONAL"],
                    "lef_files": ["${IHP130_PDK_ROOT}/lef/mem.lef"],
                    "liberty_files": ["${IHP130_PDK_ROOT}/lib/mem_typ.lib"],
                    "gds_files": ["${IHP130_PDK_ROOT}/gds/mem.gds"],
                },
                "ports": {"A_CLK": {"dir": "input", "type": "clock"}, "A_DOUT": {"dir": "output", "type": "i64"}},
                "supports": {"load": {"ii": 1, "d": 2}, "store": {"ii": 1, "d": 1}},
                "memory": {"word_t": "i64", "word_len": 64},
            },
            "GEN_ALU": {
                "kind": "combinational",
                "hdl": {"language": "verilog", "module": "GEN_ALU", "source": "gen_alu.v"},
                "ports": {"a": {"dir": "input", "type": "i32"}, "y": {"dir": "output", "type": "i32"}},
                "supports": {"add": {"ii": 1, "d": 1}},
            },
        }

        macros = collect_flow_macros("ihp130", design, component_library)
        self.assertEqual(len(macros), 2)
        by_instance = {macro.instance_name: macro for macro in macros}
        self.assertIn("A_mem_inst", by_instance)
        self.assertIn("gen_alu0", by_instance)
        self.assertEqual(by_instance["A_mem_inst"].module_name, "RM_IHPSG13_1P_64x64_c2_bm_bist")
        self.assertEqual(
            by_instance["A_mem_inst"].verilog_sources,
            ("${IHP130_PDK_ROOT}/mem.v", "${IHP130_PDK_ROOT}/core.v"),
        )
        self.assertEqual(by_instance["A_mem_inst"].lef_files, ("${IHP130_PDK_ROOT}/lef/mem.lef",))
        self.assertEqual(by_instance["A_mem_inst"].liberty_files, ("${IHP130_PDK_ROOT}/lib/mem_typ.lib",))
        self.assertEqual(by_instance["A_mem_inst"].gds_files, ("${IHP130_PDK_ROOT}/gds/mem.gds",))
        self.assertEqual(by_instance["gen_alu0"].module_name, "GEN_ALU")
        self.assertEqual(by_instance["gen_alu0"].verilog_sources, ("gen_alu.v",))


if __name__ == "__main__":
    unittest.main()
