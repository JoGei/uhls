from __future__ import annotations

import unittest
from pathlib import Path

from uhls.backend.hls.lib import (
    import_verilog_component_stub,
    import_verilog_component_stub_from_files,
    materialize_hdl_component_spec,
    resolve_component_type,
    validate_component_library,
)
from uhls.backend.hls.impl.vendor.ihp130.import_ihp_sram_lib import (
    _enrich_ihp_memory_stub,
    _rewrite_hdl_paths_with_env,
)


class ComponentLibraryImportTests(unittest.TestCase):
    """Coverage for component-library HDL import helpers."""

    def test_import_verilog_component_stub_extracts_parameters_ports_and_ops(self) -> None:
        stub = import_verilog_component_stub(
            verilog_text="""
            module SEQADD #(
              parameter WIDTH = 32,
              parameter STYLE = "fast"
            ) (
              input clk,
              input rst,
              input signed [WIDTH-1:0] a,
              input [31:0] b,
              output [31:0] y
            );
            endmodule
            """,
            module_name="SEQADD",
            source_path=Path("fu.v"),
            ops=("add",),
        )

        self.assertEqual(stub["kind"], "combinational")
        self.assertEqual(stub["hdl"]["language"], "verilog")
        self.assertEqual(stub["hdl"]["module"], "SEQADD")
        self.assertEqual(stub["hdl"]["source"], "fu.v")
        self.assertEqual(stub["parameters"]["WIDTH"]["kind"], "int")
        self.assertEqual(stub["parameters"]["WIDTH"]["default"], "32")
        self.assertEqual(stub["parameters"]["STYLE"]["kind"], "string")
        self.assertEqual(stub["ports"]["clk"], {"dir": "input", "type": "i1"})
        self.assertEqual(stub["ports"]["a"], {"dir": "input", "type": "TODO"})
        self.assertEqual(stub["ports"]["b"], {"dir": "input", "type": "u32"})
        self.assertEqual(stub["ports"]["y"], {"dir": "output", "type": "u32"})
        self.assertEqual(stub["supports"]["add"]["ii"], "TODO")
        self.assertEqual(stub["supports"]["add"]["bind"], "TODO")

    def test_import_verilog_component_stub_accepts_nonansi_module_headers(self) -> None:
        stub = import_verilog_component_stub(
            verilog_text="""
            module SRAM_MACRO (
              A_CLK,
              A_ADDR,
              A_DIN,
              A_DOUT
            );

              input A_CLK;
              input [5:0] A_ADDR;
              input [63:0] A_DIN;
              output [63:0] A_DOUT;

            endmodule
            """,
            module_name="SRAM_MACRO",
            source_path=Path("sram.v"),
        )

        self.assertEqual(stub["hdl"]["module"], "SRAM_MACRO")
        self.assertEqual(stub["ports"]["A_CLK"], {"dir": "input", "type": "i1"})
        self.assertEqual(stub["ports"]["A_ADDR"], {"dir": "input", "type": "u6"})
        self.assertEqual(stub["ports"]["A_DIN"], {"dir": "input", "type": "u64"})
        self.assertEqual(stub["ports"]["A_DOUT"], {"dir": "output", "type": "u64"})

    def test_import_verilog_component_stub_from_files_collects_recursive_related_sources(self) -> None:
        stub = import_verilog_component_stub_from_files(
            source_files=(
                (
                    Path("leaf.v"),
                    """
                    module LEAF (
                      input [31:0] a,
                      output [31:0] y
                    );
                    endmodule
                    """,
                ),
                (
                    Path("mid.v"),
                    """
                    module MID (
                      input [31:0] a,
                      output [31:0] y
                    );
                      LEAF u_leaf (
                        .a(a),
                        .y(y)
                      );
                    endmodule
                    """,
                ),
                (
                    Path("top.v"),
                    """
                    module TOP (
                      input [31:0] a,
                      output [31:0] y
                    );
                      MID u_mid (
                        .a(a),
                        .y(y)
                      );
                    endmodule
                    """,
                ),
            ),
            module_name="TOP",
        )

        self.assertEqual(stub["hdl"]["sources"], ["leaf.v", "mid.v", "top.v"])

    def test_validate_component_library_accepts_foreign_hdl_linkage(self) -> None:
        validated = validate_component_library(
            {
                "ALU": {
                    "kind": "combinational",
                    "hdl": {
                        "language": "verilog",
                        "module": "ALU",
                        "source": "rtl/alu.v",
                    },
                    "ports": {
                        "a": {"dir": "input", "type": "i32"},
                        "y": {"dir": "output", "type": "i32"},
                    },
                    "supports": {},
                }
            }
        )

        self.assertEqual(validated["ALU"]["hdl"]["language"], "verilog")
        self.assertEqual(validated["ALU"]["hdl"]["module"], "ALU")

    def test_validate_component_library_accepts_multiple_hdl_sources(self) -> None:
        validated = validate_component_library(
            {
                "SRAM": {
                    "kind": "memory",
                    "hdl": {
                        "language": "verilog",
                        "module": "SRAM",
                        "sources": ["rtl/core.v", "rtl/top.v"],
                    },
                    "memory": {
                        "word_t": "i64",
                        "word_len": 64,
                    },
                    "ports": {
                        "addr": {"dir": "input", "type": "i32"},
                        "wdata": {"dir": "input", "type": "i64"},
                        "we": {"dir": "input", "type": "i1"},
                        "rdata": {"dir": "output", "type": "i64"},
                    },
                    "supports": {
                        "load": {"ii": 1, "d": 2},
                        "store": {"ii": 1, "d": 1},
                    },
                }
            }
        )

        self.assertEqual(validated["SRAM"]["hdl"]["sources"], ["rtl/core.v", "rtl/top.v"])

    def test_validate_component_library_accepts_hdl_include_dirs_and_defines(self) -> None:
        validated = validate_component_library(
            {
                "SRAM": {
                    "kind": "memory",
                    "hdl": {
                        "language": "verilog",
                        "module": "SRAM",
                        "sources": ["rtl/core.v", "rtl/top.v"],
                        "include_dirs": ["rtl/include"],
                        "defines": ["FUNCTIONAL", "USE_BIST=1"],
                    },
                    "memory": {
                        "word_t": "i64",
                        "word_len": 64,
                    },
                    "ports": {
                        "addr": {"dir": "input", "type": "i32"},
                        "wdata": {"dir": "input", "type": "i64"},
                        "we": {"dir": "input", "type": "i1"},
                        "rdata": {"dir": "output", "type": "i64"},
                    },
                    "supports": {
                        "load": {"ii": 1, "d": 2},
                        "store": {"ii": 1, "d": 1},
                    },
                }
            }
        )

        self.assertEqual(validated["SRAM"]["hdl"]["include_dirs"], ["rtl/include"])
        self.assertEqual(validated["SRAM"]["hdl"]["defines"], ["FUNCTIONAL", "USE_BIST=1"])

    def test_validate_component_library_accepts_tied_input_ports(self) -> None:
        validated = validate_component_library(
            {
                "SRAM": {
                    "kind": "memory",
                    "ports": {
                        "clk": {"dir": "input", "type": "clock"},
                        "addr": {"dir": "input", "type": "u6"},
                        "bist_en": {"dir": "input", "type": "i1", "tie": "false"},
                        "rdata": {"dir": "output", "type": "u64"},
                    },
                    "memory": {"word_t": "i64", "word_len": 64},
                    "supports": {"load": {"ii": 1, "d": 2}},
                }
            }
        )

        self.assertEqual(validated["SRAM"]["ports"]["bist_en"]["tie"], "false")

    def test_validate_component_library_accepts_multicycle_combinational_support(self) -> None:
        validated = validate_component_library(
            {
                "DIV": {
                    "kind": "combinational",
                    "ports": {
                        "a": {"dir": "input", "type": "i32"},
                        "b": {"dir": "input", "type": "i32"},
                        "y": {"dir": "output", "type": "i32"},
                    },
                    "supports": {
                        "div": {"ii": 1, "d": 3, "bind": {"a": "operand0", "b": "operand1", "y": "result"}}
                    },
                }
            }
        )

        self.assertEqual(validated["DIV"]["kind"], "combinational")

    def test_validate_component_library_accepts_multicycle_pipelined_support(self) -> None:
        validated = validate_component_library(
            {
                "DIV": {
                    "kind": "pipelined",
                    "ports": {
                        "a": {"dir": "input", "type": "i32"},
                        "b": {"dir": "input", "type": "i32"},
                        "y": {"dir": "output", "type": "i32"},
                    },
                    "supports": {
                        "div": {"ii": 1, "d": 3, "bind": {"a": "operand0", "b": "operand1", "y": "result"}}
                    },
                }
            }
        )

        self.assertEqual(validated["DIV"]["kind"], "pipelined")

    def test_enrich_ihp_1p_memory_stub_assigns_expected_load_store_contract(self) -> None:
        component = {
            "kind": "memory",
            "hdl": {
                "language": "verilog",
                "module": "RM_IHPSG13_1P_64x64_c2_bm_bist",
                "sources": [
                    "RM_IHPSG13_1P_core_behavioral_bm_bist.v",
                    "RM_IHPSG13_1P_64x64_c2_bm_bist.v",
                ],
            },
            "ports": {
                "A_CLK": {"dir": "input", "type": "i1"},
                "A_ADDR": {"dir": "input", "type": "u6"},
                "A_DIN": {"dir": "input", "type": "u64"},
                "A_DOUT": {"dir": "output", "type": "u64"},
                "A_DLY": {"dir": "input", "type": "i1"},
                "A_MEN": {"dir": "input", "type": "i1"},
                "A_REN": {"dir": "input", "type": "i1"},
                "A_WEN": {"dir": "input", "type": "i1"},
                "A_BM": {"dir": "input", "type": "u64"},
                "A_BIST_CLK": {"dir": "input", "type": "i1"},
                "A_BIST_EN": {"dir": "input", "type": "i1"},
                "A_BIST_MEN": {"dir": "input", "type": "i1"},
                "A_BIST_WEN": {"dir": "input", "type": "i1"},
                "A_BIST_REN": {"dir": "input", "type": "i1"},
                "A_BIST_ADDR": {"dir": "input", "type": "u6"},
                "A_BIST_DIN": {"dir": "input", "type": "u64"},
                "A_BIST_BM": {"dir": "input", "type": "u64"},
            },
            "supports": {},
        }

        _enrich_ihp_memory_stub(component, "RM_IHPSG13_1P_64x64_c2_bm_bist")

        self.assertEqual(component["hdl"]["defines"], ["FUNCTIONAL"])
        self.assertEqual(
            component["parameters"],
            {
                "word_t": {"kind": "type", "required": True},
                "word_len": {"kind": "int", "required": True},
            },
        )
        self.assertEqual(component["ports"]["A_CLK"]["type"], "clock")
        self.assertEqual(component["ports"]["A_DIN"]["type"], "word_t")
        self.assertEqual(component["ports"]["A_DOUT"]["type"], "word_t")
        self.assertEqual(component["ports"]["A_BM"]["type"], "word_t")
        self.assertEqual(component["ports"]["A_BIST_CLK"]["tie"], "false")
        self.assertEqual(component["ports"]["A_BIST_EN"]["tie"], "false")
        self.assertEqual(component["ports"]["A_BIST_ADDR"]["tie"], "false")
        self.assertEqual(component["memory"], {"word_t": "i64", "word_len": 64})
        self.assertEqual(
            component["supports"]["load"],
            {
                "ii": 1,
                "d": 2,
                "mode": "read",
                "bind": {
                    "A_DLY": "true",
                    "A_MEN": "true",
                    "A_REN": "true",
                    "A_WEN": "false",
                    "A_ADDR": "operand1",
                    "A_DOUT": "result",
                },
            },
        )
        self.assertEqual(
            component["supports"]["store"],
            {
                "ii": 1,
                "d": 1,
                "mode": "write",
                "bind": {
                    "A_DLY": "true",
                    "A_MEN": "true",
                    "A_REN": "false",
                    "A_WEN": "true",
                    "A_ADDR": "operand1",
                    "A_DIN": "operand2",
                    "A_BM": "18446744073709551615:u64",
                },
            },
        )

    def test_materialize_hdl_component_spec_maps_semantic_base_type_to_verilog_width(self) -> None:
        library = validate_component_library(
            {
                "DIV": {
                    "kind": "pipelined",
                    "hdl": {
                        "language": "verilog",
                        "module": "DIV",
                        "source": "src/uhls/backend/hls/impl/generic/gen_div.v",
                        "parameters": {
                            "W": "$bits(base_t)",
                        },
                    },
                    "parameters": {
                        "base_t": {"kind": "type", "required": True},
                    },
                    "ports": {
                        "a": {"dir": "input", "type": "base_t"},
                        "div": {"dir": "output", "type": "base_t"},
                    },
                    "supports": {
                        "div": {
                            "ii": 1,
                            "d": 3,
                            "types": {"operand0": "base_t", "operand1": "base_t", "result": "base_t"},
                            "bind": {"a": "operand0", "div": "result"},
                        }
                    },
                }
            }
        )

        self.assertEqual(resolve_component_type("base_t", {"base_t": "i8"}), "i8")
        self.assertEqual(materialize_hdl_component_spec(library, "DIV<base_t=i8>"), "DIV<W=8>")
        self.assertEqual(materialize_hdl_component_spec(library, "DIV<base_t=i32>"), "DIV<W=32>")
