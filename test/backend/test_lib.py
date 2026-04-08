from __future__ import annotations

import unittest
from pathlib import Path

from uhls.backend.hls.lib import import_verilog_component_stub, validate_component_library


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
