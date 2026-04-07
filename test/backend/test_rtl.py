from __future__ import annotations

import unittest

from uhls.backend.hls import lower_fsm_to_uglir, lower_uglir_to_rtl
from uhls.backend.hls.uhir import parse_uhir
import json


class RTLLoweringTests(unittest.TestCase):
    """Coverage for uglir-to-RTL emission."""

    def test_lower_uglir_to_verilog_emits_static_module(self) -> None:
        fsm_design = parse_uhir(
            """
            design add1
            stage fsm
            schedule kind=control_steps
            input  x : i32
            output result : i32
            resources {
              fu ewms0 : EWMS
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_add1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state DONE code=16
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[ewms0<-v1]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_add1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, 1:i32 : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=ewms0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
              value v1 -> r_i32_0 live=[1:1]
            }
            """
        )

        uglir_design = lower_fsm_to_uglir(fsm_design)
        verilog = lower_uglir_to_rtl(uglir_design, hdl="verilog")

        self.assertIn("module add1 (", verilog)
        self.assertIn("input clk,", verilog)
        self.assertIn("output req_ready,", verilog)
        self.assertIn("reg [4:0] state_q;", verilog)
        self.assertIn("wire req_fire_n;", verilog)
        self.assertIn("assign req_fire_n = req_valid & req_ready;", verilog)
        self.assertIn("localparam SEL_R_I32_0_N_HOLD = 1'd0;", verilog)
        self.assertIn("assign mx_r_i32_0_n = (sel_r_i32_0_n == SEL_R_I32_0_N_HOLD) ? r_i32_0_q : ewms0_y_n;", verilog)
        self.assertIn("EWMS ewms0 (", verilog)
        self.assertIn(".go(ewms0_go_n)", verilog)
        self.assertIn(".y(ewms0_y_n)", verilog)
        self.assertIn("always @(posedge clk) begin", verilog)
        self.assertIn("if (rst) begin", verilog)
        self.assertIn("state_q <= next_state_n;", verilog)
        self.assertIn("r_i32_0_q <= mx_r_i32_0_n;", verilog)
        self.assertTrue(verilog.rstrip().endswith("endmodule"))

    def test_lower_uglir_to_verilog_emits_wishbone_slave_wrapper(self) -> None:
        fsm_design = parse_uhir(
            """
            design add1
            stage fsm
            schedule kind=control_steps
            input  x : i32
            output result : i32
            resources {
              fu ewms0 : EWMS
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_add1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state DONE code=2
              transition IDLE -> DONE when=req_valid && req_ready
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit DONE resp_valid=true
            }

            region proc_add1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = ret x class=CTRL ii=0 delay=0 start=0 end=0
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
              edge data v0 -> v1
              edge data v1 -> v2
              steps [0:0]
              latency 1
            }
            """
        )

        uglir_design = lower_fsm_to_uglir(fsm_design)
        verilog = lower_uglir_to_rtl(uglir_design, hdl="verilog", wrap="slave", protocol="wishbone")

        self.assertIn("// Wrapper for wrap=slave protocol=wishbone.", verilog)
        self.assertIn("module add1_core (", verilog)
        self.assertIn("module add1 #(", verilog)
        self.assertIn("parameter [31:0] WB_BASE_ADDR = 32'h0000_0000", verilog)
        self.assertIn("input wb_cyc_i,", verilog)
        self.assertIn("input wb_stb_i,", verilog)
        self.assertIn("output [31:0] wb_dat_o,", verilog)
        self.assertIn("localparam [31:0] WB_REG_CONTROL_STATUS = WB_BASE_ADDR + 32'h0000_0000;", verilog)
        self.assertIn("reg start_pending_q;", verilog)
        self.assertIn("reg busy_q;", verilog)
        self.assertIn("reg done_q;", verilog)
        self.assertIn("assign wb_ack_o = wb_req_n;", verilog)
        self.assertIn("assign core_req_valid_n = start_pending_q;", verilog)
        self.assertIn("assign core_resp_ready_n = 1'b1;", verilog)
        self.assertIn("if (wb_adr_i == WB_REG_CONTROL_STATUS && wb_dat_i[0]) begin", verilog)
        self.assertIn("add1_core core (", verilog)

    def test_lower_uglir_to_verilog_accepts_none_memory_wrap_pair(self) -> None:
        fsm_design = parse_uhir(
            """
            design add1
            stage fsm
            schedule kind=control_steps
            input  x : i32
            output result : i32
            resources {
              fu ewms0 : EWMS
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_add1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state DONE code=2
              transition IDLE -> DONE when=req_valid && req_ready
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit DONE resp_valid=true
            }

            region proc_add1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = ret x class=CTRL ii=0 delay=0 start=0 end=0
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
              edge data v0 -> v1
              edge data v1 -> v2
              steps [0:0]
              latency 1
            }
            """
        )

        uglir_design = lower_fsm_to_uglir(fsm_design)
        verilog = lower_uglir_to_rtl(uglir_design, hdl="verilog", wrap="none", protocol="memory")

        self.assertIn("module add1 (", verilog)
        self.assertNotIn("module add1_core (", verilog)
        self.assertIn("input req_valid,", verilog)
        self.assertIn("input resp_ready,", verilog)
        self.assertIn("output req_ready,", verilog)
        self.assertIn("output resp_valid,", verilog)

    def test_lower_uglir_to_verilog_exposes_memory_bundle_without_mem_instance(self) -> None:
        fsm_design = parse_uhir(
            """
            design read1
            stage fsm
            schedule kind=control_steps
            input  A : memref<i32, 4>
            input  i : i32
            output result : i32
            resources {
              fu mr0 : MEM
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_read1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state DONE code=16
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mr0<-v1]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_read1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = load A, i : i32 class=MEM ii=1 delay=1 start=0 end=0 bind=mr0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
              value v1 -> r_i32_0 live=[1:1]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "MEM": {
                  "kind": "memory",
                  "ports": {
                    "addr": { "dir": "input", "type": "i32" },
                    "wdata": { "dir": "input", "type": "i32" },
                    "we": { "dir": "input", "type": "i1" },
                    "rdata": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "load": {
                      "ii": 1,
                      "d": 1,
                      "bind": {
                        "addr": "operand1",
                        "rdata": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)
        verilog = lower_uglir_to_rtl(uglir_design, hdl="verilog", wrap="none", protocol="memory")

        self.assertIn("input signed [31:0] A_rdata,", verilog)
        self.assertIn("output signed [31:0] A_addr", verilog)
        self.assertIn("assign A_addr = mr0_addr_n;", verilog)
        self.assertIn("assign mr0_rdata_n = A_rdata;", verilog)
        self.assertNotIn("MEM mr0 (", verilog)

    def test_lower_uglir_to_verilog_sizes_wishbone_wrapper_memory_from_memref_extent(self) -> None:
        fsm_design = parse_uhir(
            """
            design read1
            stage fsm
            schedule kind=control_steps
            input  A : memref<i32, 4>
            input  i : i32
            output result : i32
            resources {
              fu mr0 : MEM
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_read1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state DONE code=16
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mr0<-v1]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_read1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = load A, i : i32 class=MEM ii=1 delay=1 start=0 end=0 bind=mr0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
              value v1 -> r_i32_0 live=[1:1]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "MEM": {
                  "kind": "memory",
                  "ports": {
                    "addr": { "dir": "input", "type": "i32" },
                    "wdata": { "dir": "input", "type": "i32" },
                    "we": { "dir": "input", "type": "i1" },
                    "rdata": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "load": {
                      "ii": 1,
                      "d": 1,
                      "bind": {
                        "addr": "operand1",
                        "rdata": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)
        verilog = lower_uglir_to_rtl(uglir_design, hdl="verilog", wrap="slave", protocol="wishbone")

        self.assertIn(("A_depth", 4, "u32"), [(const.name, const.value, const.type) for const in uglir_design.constants])
        self.assertIn("localparam [31:0] WB_MEM_A_BASE = WB_BASE_ADDR + 32'h0000_1000;", verilog)
        self.assertIn("assign A_bus_hit_n = wb_adr_i >= WB_MEM_A_BASE && wb_adr_i < (WB_MEM_A_BASE + 32'h0000_0010);", verilog)
        self.assertIn("reg signed [31:0] A_mem_q [0:3];", verilog)

    def test_lower_uglir_to_verilog_rejects_missing_output_driver(self) -> None:
        uglir_design = parse_uhir(
            """
            design bad
            stage uglir
            input  clk : clock
            output req_ready : i1
            resources {
              reg state : u1
              net next_state : u1
            }
            seq clk {
              state <= next_state
            }
            """
        )

        with self.assertRaisesRegex(ValueError, "output 'req_ready' must have exactly one assign driver"):
            lower_uglir_to_rtl(uglir_design, hdl="verilog")

    def test_lower_uglir_to_verilog_rejects_continuous_assign_to_reg(self) -> None:
        uglir_design = parse_uhir(
            """
            design bad
            stage uglir
            input  clk : clock
            output y : i1
            resources {
              reg r : i1
            }
            assign y = r
            assign r = true
            seq clk {
              r <= false
            }
            """
        )

        with self.assertRaisesRegex(ValueError, "assign 'r = \\.\\.\\.' must target an output port or net resource"):
            lower_uglir_to_rtl(uglir_design, hdl="verilog")

    def test_lower_uglir_to_verilog_rejects_incompatible_mux_case_type(self) -> None:
        uglir_design = parse_uhir(
            """
            design bad
            stage uglir
            input  bad_sel : ctrl
            input  x : i1
            output y : i32
            resources {
              net sel : ctrl
              mux mx : i32
            }
            assign y = mx
            assign sel = ZERO
            mux mx : i32 sel=sel {
              ZERO -> bad_sel
            }
            """
        )

        with self.assertRaisesRegex(ValueError, "mux 'mx' case 'ZERO' source 'bad_sel' has type 'ctrl', expected 'i32'"):
            lower_uglir_to_rtl(uglir_design, hdl="verilog")

    def test_lower_uglir_to_verilog_rejects_memory_word_type_mismatch(self) -> None:
        uglir_design = parse_uhir(
            """
            design bad_mem
            stage uglir
            input  clk : clock
            input  rst : i1
            input  req_valid : i1
            input  resp_ready : i1
            input  A_rdata : i32
            output req_ready : i1
            output resp_valid : i1
            output A_addr : i32
            resources {
              port A : MEM<word_t=i16,word_len=4> A
              reg state_q : u1
              net next_state_n : u1
            }
            assign req_ready = true
            assign resp_valid = false
            assign A_addr = 0:i32
            seq clk {
              state_q <= next_state_n
            }
            """
        )

        with self.assertRaisesRegex(
            ValueError,
            "memory interface 'A' declares word_t=i16 but core bundle uses i32",
        ):
            lower_uglir_to_rtl(uglir_design, hdl="verilog", wrap="slave", protocol="wishbone")
