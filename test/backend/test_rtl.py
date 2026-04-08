from __future__ import annotations

import unittest

from uhls.backend.hls import lower_fsm_to_uglir, lower_uglir_to_rtl, wrap_uglir_design
from uhls.backend.hls.uglir import parse_uglir
from uhls.backend.hls.uhir import parse_uhir
import json


class RTLLoweringTests(unittest.TestCase):
    """Coverage for uglir-to-RTL emission."""

    @staticmethod
    def _vendor_memory_component_library() -> dict[str, dict[str, object]]:
        return {
            "VENDOR_MEM": {
                "kind": "memory",
                "parameters": {
                    "word_t": {"kind": "type", "required": True},
                    "word_len": {"kind": "int", "required": True},
                },
                "hdl": {
                    "language": "verilog",
                    "module": "VENDOR_MEM",
                },
                "ports": {
                    "A_CLK": {"dir": "input", "type": "clock"},
                    "A_MEN": {"dir": "input", "type": "i1"},
                    "A_WEN": {"dir": "input", "type": "i1"},
                    "A_REN": {"dir": "input", "type": "i1"},
                    "A_ADDR": {"dir": "input", "type": "u2"},
                    "A_DIN": {"dir": "input", "type": "word_t"},
                    "A_DLY": {"dir": "input", "type": "i1"},
                    "A_DOUT": {"dir": "output", "type": "word_t"},
                },
                "supports": {
                    "load": {
                        "ii": 1,
                        "d": 2,
                        "bind": {
                            "A_DLY": "true",
                            "A_MEN": "true",
                            "A_REN": "true",
                            "A_WEN": "false",
                            "A_ADDR": "operand1",
                            "A_DOUT": "result",
                        },
                    },
                    "store": {
                        "ii": 1,
                        "d": 1,
                        "bind": {
                            "A_DLY": "true",
                            "A_MEN": "true",
                            "A_REN": "false",
                            "A_WEN": "true",
                            "A_ADDR": "operand1",
                            "A_DIN": "operand2",
                        },
                    },
                },
            }
        }

    @staticmethod
    def _generic_ff_memory_component_library() -> dict[str, dict[str, object]]:
        return {
            "GEN_FF_MEM": {
                "kind": "memory",
                "hdl": {
                    "language": "verilog",
                    "module": "GEN_FF_MEM",
                    "parameters": {
                        "W": "$bits(word_t)",
                        "DEPTH": "word_len",
                    },
                },
                "parameters": {
                    "word_t": {"kind": "type", "required": True},
                    "word_len": {"kind": "int", "required": True},
                },
                "ports": {
                    "clk": {"dir": "input", "type": "clock"},
                    "addr": {"dir": "input", "type": "i16"},
                    "wdata": {"dir": "input", "type": "word_t"},
                    "we": {"dir": "input", "type": "i1"},
                    "rdata": {"dir": "output", "type": "word_t"},
                },
                "supports": {
                    "load": {"ii": 1, "d": 1, "bind": {"addr": "operand1", "rdata": "result", "we": "false"}},
                    "store": {"ii": 1, "d": 1, "bind": {"addr": "operand1", "wdata": "operand2", "we": "true"}},
                },
            }
        }

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

    def test_lower_uglir_to_verilog_supports_async_active_low_reset(self) -> None:
        uglir_design = parse_uglir(
            """
            design add1
            stage uglir
            input  clk : clock
            input  rst : i1
            output y : i1
            resources {
              reg state_q : i1
            }
            assign y = state_q
            seq clk {
              if rst {
                state_q <= false
              } else {
                state_q <= true
              }
            }
            """
        )

        verilog = lower_uglir_to_rtl(uglir_design, hdl="verilog", reset="async+active_lo")

        self.assertIn("always @(posedge clk or negedge rst) begin", verilog)
        self.assertIn("if (!(rst)) begin", verilog)

    def test_lower_uglir_to_verilog_emits_parameterized_instance_overrides(self) -> None:
        uglir_design = parse_uglir(
            """
            design typed_div
            stage uglir
            input  clk : clock
            input  rst : i1
            input  a : i8
            input  b : i8
            output y : i8
            resources {
              net div0_div_n : i8
              inst div0 : DIV<W=8>
            }
            div0.clk(clk)
            div0.rst(rst)
            div0.a(a)
            div0.b(b)
            div0.div(div0_div_n)
            assign y = div0_div_n
            seq clk {
            }
            """
        )

        verilog = lower_uglir_to_rtl(uglir_design, hdl="verilog")

        self.assertIn("DIV #(", verilog)
        self.assertIn(".W(8)", verilog)
        self.assertIn(") div0 (", verilog)

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
        wrapped_uglir = wrap_uglir_design(uglir_design, wrap="slave", protocol="wishbone")
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        self.assertEqual(len(wrapped_uglir.address_maps), 1)
        self.assertEqual(wrapped_uglir.address_maps[0].name, "wishbone")
        self.assertEqual(wrapped_uglir.address_maps[0].entries[0].name, "control_status")
        self.assertEqual(wrapped_uglir.address_maps[0].entries[1].attributes["offset"], "32'h0000_0100")
        self.assertEqual(wrapped_uglir.address_maps[0].entries[2].attributes["access"], "ro")
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
        self.assertIn("assign req_valid = start_pending_q;", verilog)
        self.assertIn("assign resp_ready = 1'b1;", verilog)
        self.assertIn(
            "start_pending_q <= ((wb_req_n && wb_we_i) && (wb_adr_i == WB_REG_CONTROL_STATUS) && wb_dat_i[0] && wb_sel_i[0]) ? 1'b1",
            verilog,
        )
        self.assertNotIn("wb_err_o", verilog)

    def test_lower_uglir_to_verilog_instantiates_selected_wishbone_wrapper_memory(self) -> None:
        uglir_design = parse_uglir(
            """
            design memcore
            stage uglir
            input  clk : clock
            input  rst : i1
            input  req_valid : i1
            input  resp_ready : i1
            input  A_rdata : i32
            output req_ready : i1
            output resp_valid : i1
            output A_addr : u2
            output A_wdata : i32
            output A_we : i1
            resources {
              port A : VENDOR_MEM<word_t=i32,word_len=4> A
            }
            assign req_ready = true
            assign resp_valid = false
            assign A_addr = 0:u2
            assign A_wdata = 0:i32
            assign A_we = false
            """
        )

        wrapped_uglir = wrap_uglir_design(
            uglir_design,
            wrap="slave",
            protocol="wishbone",
            component_library=self._vendor_memory_component_library(),
        )
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        self.assertIn("inst A_mem_inst : VENDOR_MEM", "\n".join(f"{r.kind} {r.id} : {r.value}" for r in wrapped_uglir.resources))
        self.assertNotIn("mem A_mem_q", "\n".join(f"{r.kind} {r.id} : {r.value}" for r in wrapped_uglir.resources))
        self.assertIn("A_mem_inst.A_CLK(clk)", "\n".join(f"{a.instance}.{a.port}({a.signal})" for a in wrapped_uglir.attachments))
        self.assertIn("VENDOR_MEM A_mem_inst (", verilog)
        self.assertIn("assign A_rdata = A_mem_rdata_n;", verilog)
        self.assertIn("reg A_bus_read_pending_q;", verilog)

    def test_lower_uglir_to_verilog_instantiates_selected_obi_wrapper_memory(self) -> None:
        uglir_design = parse_uglir(
            """
            design memcore
            stage uglir
            input  clk : clock
            input  rst : i1
            input  req_valid : i1
            input  resp_ready : i1
            input  A_rdata : i32
            output req_ready : i1
            output resp_valid : i1
            output A_addr : u2
            output A_wdata : i32
            output A_we : i1
            resources {
              port A : VENDOR_MEM<word_t=i32,word_len=4> A
            }
            assign req_ready = true
            assign resp_valid = false
            assign A_addr = 0:u2
            assign A_wdata = 0:i32
            assign A_we = false
            """
        )

        wrapped_uglir = wrap_uglir_design(
            uglir_design,
            wrap="slave",
            protocol="obi",
            component_library=self._vendor_memory_component_library(),
        )
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        self.assertIn("inst A_mem_inst : VENDOR_MEM", "\n".join(f"{r.kind} {r.id} : {r.value}" for r in wrapped_uglir.resources))
        self.assertNotIn("mem A_mem_q", "\n".join(f"{r.kind} {r.id} : {r.value}" for r in wrapped_uglir.resources))
        self.assertIn("VENDOR_MEM A_mem_inst (", verilog)
        self.assertIn("assign obi_rvalid_o = obi_rsp_pending_q | (A_bus_read_pending_q);", verilog)
        self.assertIn("assign A_rdata = A_mem_rdata_n;", verilog)

    def test_lower_uglir_to_verilog_instantiates_selected_generic_ff_wrapper_memory(self) -> None:
        uglir_design = parse_uglir(
            """
            design memcore
            stage uglir
            input  clk : clock
            input  rst : i1
            input  req_valid : i1
            input  resp_ready : i1
            input  A_rdata : i16
            output req_ready : i1
            output resp_valid : i1
            output A_addr : i16
            output A_wdata : i16
            output A_we : i1
            resources {
              port A : GEN_FF_MEM<word_t=i16,word_len=4> A
            }
            assign req_ready = true
            assign resp_valid = false
            assign A_addr = 0:i16
            assign A_wdata = 0:i16
            assign A_we = false
            """
        )

        wrapped_uglir = wrap_uglir_design(
            uglir_design,
            wrap="slave",
            protocol="wishbone",
            component_library=self._generic_ff_memory_component_library(),
        )
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        resource_dump = "\n".join(f"{r.kind} {r.id} : {r.value}" for r in wrapped_uglir.resources)
        self.assertIn("inst A_mem_inst : GEN_FF_MEM<", resource_dump)
        self.assertIn("DEPTH=4", resource_dump)
        self.assertIn("W=16", resource_dump)
        self.assertNotIn("mem A_mem_q", "\n".join(f"{r.kind} {r.id} : {r.value}" for r in wrapped_uglir.resources))
        self.assertIn("GEN_FF_MEM #(", verilog)
        self.assertIn(".W(16)", verilog)
        self.assertIn(".DEPTH(4)", verilog)

    def test_lower_uglir_to_verilog_emits_wishbone_err_wrapper(self) -> None:
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
        wrapped_uglir = wrap_uglir_design(uglir_design, wrap="slave", protocol="wishbone+err")
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        self.assertEqual(len(wrapped_uglir.address_maps), 1)
        self.assertEqual(wrapped_uglir.address_maps[0].entries[0].attributes["symbol"], "WB_REG_CONTROL_STATUS")
        self.assertIn("output wb_err_o", verilog)
        self.assertIn("wire wb_hit_n;", verilog)
        self.assertIn("assign wb_ack_o = wb_req_n & wb_hit_n;", verilog)
        self.assertIn("assign wb_err_o = wb_req_n & !wb_hit_n;", verilog)

    def test_lower_uglir_to_verilog_emits_obi_slave_wrapper(self) -> None:
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
        wrapped_uglir = wrap_uglir_design(uglir_design, wrap="slave", protocol="obi")
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        self.assertEqual(len(wrapped_uglir.address_maps), 1)
        self.assertEqual(wrapped_uglir.address_maps[0].name, "obi")
        self.assertEqual(wrapped_uglir.address_maps[0].entries[0].name, "control_status")
        self.assertEqual(wrapped_uglir.address_maps[0].entries[1].attributes["offset"], "32'h0000_0100")
        self.assertEqual(wrapped_uglir.address_maps[0].entries[2].attributes["access"], "ro")
        self.assertIn("module add1 #(", verilog)
        self.assertIn("parameter [31:0] OBI_BASE_ADDR = 32'h0000_0000", verilog)
        self.assertIn("input obi_req_i,", verilog)
        self.assertIn("input [31:0] obi_addr_i,", verilog)
        self.assertIn("input obi_rready_i,", verilog)
        self.assertIn("output obi_gnt_o,", verilog)
        self.assertIn("output obi_rvalid_o,", verilog)
        self.assertIn("output [31:0] obi_rdata_o", verilog)
        self.assertIn("localparam [31:0] OBI_REG_CONTROL_STATUS = OBI_BASE_ADDR + 32'h0000_0000;", verilog)
        self.assertIn("reg obi_rsp_pending_q;", verilog)
        self.assertIn("reg [31:0] obi_rsp_rdata_q;", verilog)
        self.assertIn("assign obi_gnt_o = obi_req_i & !obi_rsp_pending_q;", verilog)
        self.assertIn("assign obi_rvalid_o = obi_rsp_pending_q;", verilog)
        self.assertIn("assign obi_rdata_o = obi_rsp_rdata_q;", verilog)
        self.assertIn("assign req_valid = start_pending_q;", verilog)
        self.assertIn("assign resp_ready = 1'b1;", verilog)

    def test_lower_uglir_to_verilog_emits_obi_burst_slave_wrapper(self) -> None:
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
        wrapped_uglir = wrap_uglir_design(uglir_design, wrap="slave", protocol="obi+burst")
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        self.assertEqual(len(wrapped_uglir.address_maps), 1)
        self.assertEqual(wrapped_uglir.address_maps[0].name, "obi")
        self.assertIn(("OBI_REG_CONTROL_STATUS", "OBI_BASE_ADDR + 32'h0000_0000", "u32"), [(const.name, const.value, const.type) for const in wrapped_uglir.constants])
        self.assertIn("mem obi_rsp_fifo_q : u32[2]", "\n".join(
            f"{resource.kind} {resource.id} : {resource.value}"
            for resource in wrapped_uglir.resources
        ))
        self.assertIn("reg obi_rsp_count_q : u2", "\n".join(
            f"{resource.kind} {resource.id} : {resource.value}"
            for resource in wrapped_uglir.resources
        ))
        self.assertIn("assign obi_gnt_o = obi_req_i & !(obi_rsp_count_q == 2:u2)", "\n".join(
            f"assign {assign.target} = {assign.expr}" for assign in wrapped_uglir.assigns
        ))
        self.assertIn("reg [31:0] obi_rsp_fifo_q [0:1];", verilog)
        self.assertIn("reg obi_rsp_head_q;", verilog)
        self.assertIn("reg obi_rsp_tail_q;", verilog)
        self.assertIn("reg [1:0] obi_rsp_count_q;", verilog)
        self.assertIn("assign obi_gnt_o = obi_req_i & !(obi_rsp_count_q == 2'd2);", verilog)
        self.assertIn("assign obi_rvalid_o = obi_rsp_count_q != 2'd0;", verilog)
        self.assertIn("assign obi_rdata_o = obi_rsp_fifo_q[obi_rsp_head_q];", verilog)

    def test_lower_uglir_to_verilog_applies_wishbone_byte_selects_to_scalar_inputs(self) -> None:
        fsm_design = parse_uhir(
            """
            design passthrough
            stage fsm
            schedule kind=control_steps
            input  x : i16
            output result : i16
            resources {
              fu ctrl0 : CTRL
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_passthrough {
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

            region proc_passthrough kind=procedure {
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
        wrapped_uglir = wrap_uglir_design(uglir_design, wrap="slave", protocol="wishbone")
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        self.assertIn(
            "x_q <= {(wb_sel_i[1] ? wb_dat_i[15:8] : x_q[15:8]), (wb_sel_i[0] ? wb_dat_i[7:0] : x_q[7:0])};",
            verilog,
        )

    def test_lower_uglir_to_verilog_applies_wishbone_byte_selects_to_memory_windows(self) -> None:
        uglir_design = parse_uglir(
            """
            design mem_core
            stage uglir
            input  clk : clock
            input  rst : i1
            input  req_valid : i1
            input  resp_ready : i1
            input  A_rdata : i16
            output req_ready : i1
            output resp_valid : i1
            output A_addr : i16
            resources {
              port A : MEM<word_t=i16,word_len=4> A
              reg state_q : u1
              net next_state_n : u1
            }
            assign req_ready = true
            assign resp_valid = false
            assign A_addr = 0:i16
            assign next_state_n = 0:u1
            seq clk {
              state_q <= next_state_n
            }
            """
        )

        wrapped_uglir = wrap_uglir_design(uglir_design, wrap="slave", protocol="wishbone")
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        self.assertIn(
            "A_mem_q[A_bus_word_addr_n] <= {(wb_sel_i[1] ? wb_dat_i[15:8] : A_mem_q[A_bus_word_addr_n][15:8]), (wb_sel_i[0] ? wb_dat_i[7:0] : A_mem_q[A_bus_word_addr_n][7:0])};",
            verilog,
        )

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
        wrapped_uglir = wrap_uglir_design(uglir_design, wrap="none", protocol="memory")
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

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
        wrapped_uglir = wrap_uglir_design(uglir_design, wrap="none", protocol="memory")
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        self.assertIn("input signed [31:0] A_rdata,", verilog)
        self.assertIn("output signed [31:0] A_addr", verilog)
        self.assertIn("assign A_addr = mr0_addr_q;", verilog)
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
        wrapped_uglir = wrap_uglir_design(uglir_design, wrap="slave", protocol="wishbone")
        verilog = lower_uglir_to_rtl(wrapped_uglir, hdl="verilog")

        self.assertIn(("A_depth", 4, "u32"), [(const.name, const.value, const.type) for const in uglir_design.constants])
        self.assertIn("localparam [31:0] WB_MEM_A_BASE = WB_BASE_ADDR + 32'h0000_1000;", verilog)
        self.assertIn("assign A_bus_hit_n = wb_adr_i >= WB_MEM_A_BASE && wb_adr_i < (WB_MEM_A_BASE + 32'h0000_0010);", verilog)
        self.assertIn("reg signed [31:0] A_mem_q [0:3];", verilog)

    def test_lower_uglir_to_verilog_rejects_missing_output_driver(self) -> None:
        uglir_design = parse_uglir(
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
        uglir_design = parse_uglir(
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
        uglir_design = parse_uglir(
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
        uglir_design = parse_uglir(
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
            wrap_uglir_design(uglir_design, wrap="slave", protocol="wishbone")

    def test_wrap_uglir_design_rejects_unsupported_protocol_feature(self) -> None:
        uglir_design = parse_uglir(
            """
            design add1
            stage uglir
            input  clk : clock
            input  rst : i1
            input  req_valid : i1
            input  resp_ready : i1
            input  x : i32
            output req_ready : i1
            output resp_valid : i1
            output result : i32
            resources {
              reg state_q : u1
              net next_state_n : u1
            }
            assign req_ready = true
            assign resp_valid = false
            assign result = x
            assign next_state_n = 0:u1
            seq clk {
              state_q <= next_state_n
            }
            """
        )

        with self.assertRaisesRegex(ValueError, "unsupported protocol feature\\(s\\) for 'wishbone': foo"):
            wrap_uglir_design(uglir_design, wrap="slave", protocol="wishbone+foo")
