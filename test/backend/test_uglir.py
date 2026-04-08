from __future__ import annotations

import json
import unittest

from uhls.backend.hls import lower_fsm_to_uglir
from uhls.backend.hls.uglir import format_uglir, parse_uglir
from uhls.backend.hls.uhir import parse_uhir


class UGLIRLoweringTests(unittest.TestCase):
    """Coverage for fsm-to-uglir lowering."""

    def test_lower_fsm_to_uglir_emits_static_glue_shell(self) -> None:
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

        self.assertEqual(uglir_design.stage, "uglir")
        self.assertEqual([(port.name, port.type) for port in uglir_design.inputs[:4]], [("clk", "clock"), ("rst", "i1"), ("req_valid", "i1"), ("resp_ready", "i1")])
        self.assertEqual([(port.name, port.type) for port in uglir_design.outputs[:3]], [("req_ready", "i1"), ("resp_valid", "i1"), ("result", "i32")])
        resource_ids = [resource.id for resource in uglir_design.resources]
        self.assertIn("state_q", resource_ids)
        self.assertIn("next_state_n", resource_ids)
        self.assertIn("ewms0", resource_ids)
        self.assertIn("ewms0_go_n", resource_ids)
        self.assertIn("ewms0_y_n", resource_ids)
        self.assertIn("mx_r_i32_0_n", resource_ids)
        self.assertEqual([(assign.target, assign.expr) for assign in uglir_design.assigns[:5]], [
            ("req_fire_n", "req_valid & req_ready"),
            ("resp_fire_n", "resp_valid & resp_ready"),
            ("req_ready", "state_q == 1"),
            ("resp_valid", "state_q == 16"),
            ("next_state_n", "(state_q == 1 && req_fire_n) ? 2 : state_q == 2 ? 4 : state_q == 4 ? 8 : state_q == 8 ? 16 : (state_q == 16 && resp_fire_n) ? 1 : 1"),
        ])
        self.assertIn(("ewms0", "go", "ewms0_go_n"), [(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments])
        self.assertIn(("ewms0", "y", "ewms0_y_n"), [(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments])
        mux = uglir_design.muxes[0]
        self.assertEqual(mux.name, "mx_r_i32_0_n")
        self.assertEqual(mux.select, "sel_r_i32_0_n")
        self.assertEqual([(case.key, case.source) for case in mux.cases], [("HOLD", "r_i32_0_q"), ("SRC_EWMS0_Y", "ewms0_y_n")])
        seq_block = uglir_design.seq_blocks[0]
        self.assertEqual(seq_block.clock, "clk")
        self.assertEqual(seq_block.reset, "rst")
        self.assertEqual([(update.target, update.value, update.enable) for update in seq_block.reset_updates], [("state_q", "1", None)])
        self.assertEqual([(update.target, update.value, update.enable) for update in seq_block.updates], [("state_q", "next_state_n", None), ("r_i32_0_q", "mx_r_i32_0_n", "latch_r_i32_0_n")])
        rendered = format_uglir(uglir_design)
        self.assertIn("stage uglir", rendered)
        self.assertIn("input  req_valid : i1", rendered)
        self.assertIn("output req_ready : i1", rendered)
        self.assertIn("inst ewms0 : EWMS", rendered)
        self.assertIn("assign req_ready = state_q == 1", rendered)
        self.assertIn("ewms0.go(ewms0_go_n)", rendered)
        self.assertIn("mux mx_r_i32_0_n : i32 sel=sel_r_i32_0_n {", rendered)
        self.assertIn("seq clk {", rendered)

    def test_lower_fsm_to_uglir_uses_component_library_ports(self) -> None:
        fsm_design = parse_uhir(
            """
            design add1
            stage fsm
            schedule kind=control_steps
            input  x : i32
            output result : i32
            resources {
              fu ewms0 : ALU
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
        component_library = json.loads(
            """
            {
              "components": {
                "ALU": {
                  "kind": "combinational",
                  "ports": {
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "op": { "dir": "input", "type": "u5" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "add": {
                      "ii": 1,
                      "d": 1,
                      "opcode": 0,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        resource_ids = [resource.id for resource in uglir_design.resources]
        self.assertIn("ewms0_a_q", resource_ids)
        self.assertIn("ewms0_b_q", resource_ids)
        self.assertIn("ewms0_op_q", resource_ids)
        self.assertIn("ewms0_y_n", resource_ids)
        self.assertIn("sel_ewms0_a_n", resource_ids)
        self.assertIn("sel_ewms0_b_n", resource_ids)
        self.assertIn("mx_ewms0_a_n", resource_ids)
        self.assertIn("mx_ewms0_b_n", resource_ids)
        self.assertNotIn("ewms0_go_n", resource_ids)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertIn(("ewms0", "a", "ewms0_a_q"), attachments)
        self.assertIn(("ewms0", "b", "ewms0_b_q"), attachments)
        self.assertIn(("ewms0", "op", "ewms0_op_q"), attachments)
        self.assertIn(("ewms0", "y", "ewms0_y_n"), attachments)
        assign_pairs = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("sel_ewms0_a_n", "state_q == 2 ? SRC_X : ZERO"), assign_pairs)
        self.assertIn(("sel_ewms0_b_n", "state_q == 2 ? CONST_1_I32 : ZERO"), assign_pairs)
        self.assertIn(("sel_ewms0_op_n", "state_q == 2 ? SRC_0 : ZERO"), assign_pairs)
        mux_a = next(mux for mux in uglir_design.muxes if mux.name == "mx_ewms0_a_n")
        mux_b = next(mux for mux in uglir_design.muxes if mux.name == "mx_ewms0_b_n")
        self.assertEqual([(case.key, case.source) for case in mux_a.cases], [("ZERO", "const_i32_0_n"), ("SRC_X", "x")])
        self.assertEqual([(case.key, case.source) for case in mux_b.cases], [("ZERO", "const_i32_0_n"), ("CONST_1_I32", "const_i32_1_n")])
        rendered = format_uglir(uglir_design)
        self.assertIn("ewms0.a(ewms0_a_q)", rendered)
        self.assertIn("ewms0.b(ewms0_b_q)", rendered)
        self.assertIn("mux mx_ewms0_a_n : i32 sel=sel_ewms0_a_n {", rendered)
        self.assertIn("ewms0.op(ewms0_op_q)", rendered)
        self.assertNotIn("ewms0.go(", rendered)

    def test_lower_fsm_to_uglir_uses_component_issue_and_result_bindings(self) -> None:
        fsm_design = parse_uhir(
            """
            design seqadd1
            stage fsm
            schedule kind=control_steps
            input  x : i32
            output result : i32
            resources {
              fu seq0 : SEQADD
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_seqadd1 {
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
              emit T0 issue=[seq0<-v1]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_seqadd1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, 1:i32 : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=seq0
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
                "SEQADD": {
                  "kind": "sequential",
                  "issue": {
                    "start": "true"
                  },
                  "ports": {
                    "start": { "dir": "input", "type": "i1" },
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "out": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "add": {
                      "ii": 1,
                      "d": 1,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "out": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        resource_ids = [resource.id for resource in uglir_design.resources]
        self.assertIn("seq0_start_n", resource_ids)
        self.assertIn("seq0_a_n", resource_ids)
        self.assertIn("seq0_b_n", resource_ids)
        self.assertIn("seq0_out_n", resource_ids)
        self.assertNotIn("seq0_go_n", resource_ids)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertIn(("seq0", "start", "seq0_start_n"), attachments)
        self.assertIn(("seq0", "out", "seq0_out_n"), attachments)
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("seq0_start_n", "state_q == 2"), assigns)
        self.assertIn(("sel_seq0_a_n", "state_q == 2 ? SRC_X : ZERO"), assigns)
        self.assertIn(("sel_seq0_b_n", "state_q == 2 ? CONST_1_I32 : ZERO"), assigns)
        mux_a = next(mux for mux in uglir_design.muxes if mux.name == "mx_seq0_a_n")
        mux_b = next(mux for mux in uglir_design.muxes if mux.name == "mx_seq0_b_n")
        self.assertEqual([(case.key, case.source) for case in mux_a.cases], [("ZERO", "const_i32_0_n"), ("SRC_X", "x")])
        self.assertEqual([(case.key, case.source) for case in mux_b.cases], [("ZERO", "const_i32_0_n"), ("CONST_1_I32", "const_i32_1_n")])
        mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_r_i32_0_n")
        self.assertIn(("SRC_SEQ0_OUT", "seq0_out_n"), [(case.key, case.source) for case in mux.cases])
        rendered = format_uglir(uglir_design)
        self.assertIn("assign seq0_start_n = state_q == 2", rendered)
        self.assertIn("assign seq0_a_n = mx_seq0_a_n", rendered)
        self.assertIn("assign seq0_b_n = mx_seq0_b_n", rendered)
        self.assertIn("seq0.a(seq0_a_n)", rendered)
        self.assertIn("seq0.b(seq0_b_n)", rendered)
        self.assertIn("seq0.start(seq0_start_n)", rendered)
        self.assertIn("seq0.out(seq0_out_n)", rendered)
        self.assertNotIn("seq0.go(", rendered)

    def test_lower_fsm_to_uglir_captures_pipelined_result_before_first_consumer_cycle(self) -> None:
        fsm_design = parse_uhir(
            """
            design pipe_use
            stage fsm
            schedule kind=control_steps
            input  x : i32
            input  y : i32
            output result : i32
            resources {
              fu mul0 : PMUL
              fu alu0 : ALU
              reg r_i32_0 : i32
              reg r_i32_1 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_pipe_use {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state T3 code=16
              state DONE code=32
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> T3
              transition T3 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mul0<-v1]
              emit T1 latch=[r_i32_0]
              emit T2 issue=[alu0<-v2]
              emit T3 latch=[r_i32_1]
              emit DONE resp_valid=true
            }

            region proc_pipe_use kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = mul x, y : i32 class=PMUL ii=1 delay=2 start=0 end=1 bind=mul0
              node v2 = add v1, 1:i32 : i32 class=ALU ii=1 delay=1 start=2 end=2 bind=alu0
              node v3 = ret v2 class=CTRL ii=0 delay=0 start=3 end=3
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
              steps [0:3]
              latency 4
              value v1 -> r_i32_0 live=[2:2]
              value v2 -> r_i32_1 live=[3:3]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "PMUL": {
                  "kind": "pipelined",
                  "ports": {
                    "clk": { "dir": "input", "type": "clock" },
                    "rst": { "dir": "input", "type": "reset", "kind": "sync", "active": "hi" },
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "mul": {
                      "ii": 1,
                      "d": 2,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                },
                "ALU": {
                  "kind": "combinational",
                  "ports": {
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "op": { "dir": "input", "type": "u5" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "add": {
                      "ii": 1,
                      "d": 1,
                      "opcode": 0,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        assign_pairs = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("latch_r_i32_0_n", "state_q == 4"), assign_pairs)
        self.assertIn(("sel_r_i32_0_n", "state_q == 4 ? SRC_MUL0_Y : HOLD"), assign_pairs)
        self.assertIn(("sel_alu0_a_n", "state_q == 8 ? SRC_R_I32_0 : ZERO"), assign_pairs)

        mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_alu0_a_n")
        self.assertEqual(
            [(case.key, case.source) for case in mux.cases],
            [("ZERO", "const_i32_0_n"), ("SRC_R_I32_0", "r_i32_0_q")],
        )

        rendered = format_uglir(uglir_design)
        self.assertIn("assign latch_r_i32_0_n = state_q == 4", rendered)
        self.assertIn("assign sel_r_i32_0_n = state_q == 4 ? SRC_MUL0_Y : HOLD", rendered)
        self.assertIn("assign sel_alu0_a_n = state_q == 8 ? SRC_R_I32_0 : ZERO", rendered)

    def test_lower_fsm_to_uglir_lowers_recursive_controller_links_to_nets(self) -> None:
        fsm_design = parse_uhir(
            """
            design dyn_call
            stage fsm
            schedule kind=hierarchical
            resources {
              fu fu_add0 : FU_ADD
            }
            controller C0 encoding=binary protocol=req_resp completion_order=in_order overlap=true region=proc_dyn_call {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=0
              state P0 code=1
              state WAIT_v1 code=2
              state DONE code=3
              transition IDLE -> P0 when=req_valid && req_ready && symb_ready_v1
              transition P0 -> WAIT_v1
              transition WAIT_v1 -> DONE when=symb_done_v1
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit P0 activate=[v1]
              emit DONE resp_valid=true
              link C_callee via=v1 act=[activate, act_valid] ready=[symb_ready_v1, act_ready] done=[symb_done_v1, done_valid] done_ready=[resp_ready, done_ready]
            }

            controller C_callee encoding=binary protocol=act_done completion_order=in_order overlap=true region=callee parent_node=v1 {
              input  act_valid : i1
              input  done_ready : i1
              output act_ready : i1
              output done_valid : i1
              state IDLE code=0
              state T0 code=1
              state DONE code=2
              transition IDLE -> T0 when=act_valid && act_ready
              transition T0 -> DONE
              transition DONE -> IDLE when=done_valid && done_ready
              emit IDLE act_ready=true
              emit DONE done_valid=true
            }

            region proc_dyn_call kind=procedure {
              region_ref callee
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = call child=callee class=CTRL ii=1 delay=1 start=0 end=0 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
              edge data v0 -> v1
              edge data v1 -> v2
              edge seq v1 -> callee hierarchy=true
              edge seq callee -> v1 hierarchy=true
            }

            region callee kind=procedure parent=proc_dyn_call {
              node c0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node c1 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
              edge data c0 -> c1
            }
            """
        )

        uglir_design = lower_fsm_to_uglir(fsm_design)

        resource_ids = [resource.id for resource in uglir_design.resources]
        self.assertIn("state_q", resource_ids)
        self.assertIn("next_state_n", resource_ids)
        self.assertIn("C_callee_state_q", resource_ids)
        self.assertIn("C_callee_next_state_n", resource_ids)
        self.assertIn("C_callee_act_valid_n", resource_ids)
        self.assertIn("C_callee_done_ready_n", resource_ids)
        self.assertIn("C_callee_act_ready_n", resource_ids)
        self.assertIn("C_callee_done_valid_n", resource_ids)
        self.assertIn("symb_ready_v1_n", resource_ids)
        self.assertIn("symb_done_v1_n", resource_ids)
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("C_callee_act_valid_n", "state_q == 1"), assigns)
        self.assertIn(("C_callee_done_ready_n", "resp_ready"), assigns)
        self.assertIn(("symb_ready_v1_n", "C_callee_act_ready_n"), assigns)
        self.assertIn(("symb_done_v1_n", "C_callee_done_valid_n"), assigns)
        self.assertIn(("C_callee_act_ready_n", "C_callee_state_q == 0"), assigns)
        self.assertIn(("C_callee_done_valid_n", "C_callee_state_q == 2"), assigns)
        rendered = format_uglir(uglir_design)
        self.assertIn("reg C_callee_state_q : u2", rendered)
        self.assertIn("net C_callee_act_valid_n : i1", rendered)
        self.assertIn("assign symb_done_v1_n = C_callee_done_valid_n", rendered)

    def test_lower_fsm_to_uglir_expands_memref_interface_for_memory_component(self) -> None:
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

        self.assertEqual([(port.name, port.type) for port in uglir_design.inputs[:6]], [("clk", "clock"), ("rst", "i1"), ("req_valid", "i1"), ("resp_ready", "i1"), ("A_rdata", "i32"), ("i", "i32")])
        self.assertEqual([(port.name, port.type) for port in uglir_design.outputs[:4]], [("req_ready", "i1"), ("resp_valid", "i1"), ("result", "i32"), ("A_addr", "i32")])
        self.assertIn(
            ("port", "A", "MEM<word_t=i32,word_len=4>", "A"),
            [(resource.kind, resource.id, resource.value, resource.target) for resource in uglir_design.resources],
        )
        self.assertNotIn(("inst", "mr0", "MEM", None), [(resource.kind, resource.id, resource.value, resource.target) for resource in uglir_design.resources])
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("sel_mr0_addr_n", "state_q == 2 ? SRC_I : ZERO"), assigns)
        self.assertIn(("A_addr", "mr0_addr_q"), assigns)
        self.assertIn(("mr0_rdata_n", "A_rdata"), assigns)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertNotIn(("mr0", "addr", "mr0_addr_n"), attachments)
        self.assertNotIn(("mr0", "rdata", "mr0_rdata_n"), attachments)
        addr_mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_mr0_addr_n")
        self.assertEqual([(case.key, case.source) for case in addr_mux.cases], [("ZERO", "const_i32_0_n"), ("SRC_I", "i")])
        rendered = format_uglir(uglir_design)
        self.assertIn("input  A_rdata : i32", rendered)
        self.assertIn("output A_addr : i32", rendered)
        self.assertIn("port A : MEM<word_t=i32,word_len=4> A", rendered)
        self.assertIn("assign mr0_rdata_n = A_rdata", rendered)
        self.assertIn("assign A_addr = mr0_addr_q", rendered)

    def test_lower_fsm_to_uglir_expands_store_memref_outputs_for_memory_component(self) -> None:
        fsm_design = parse_uhir(
            """
            design write1
            stage fsm
            schedule kind=control_steps
            input  C : memref<i32>
            input  i : i32
            input  x : i32
            resources {
              fu mw0 : MEM
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_write1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state DONE code=8
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mw0<-v1]
              emit DONE resp_valid=true
            }

            region proc_write1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = store C, i, x class=MEM ii=1 delay=1 start=0 end=0 bind=mw0
              node v2 = ret 0:i32 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
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
                    "store": {
                      "ii": 1,
                      "d": 1,
                      "mode": "write",
                      "bind": {
                        "addr": "operand1",
                        "wdata": "operand2",
                        "we": "true"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        self.assertEqual(
            [(port.name, port.type) for port in uglir_design.outputs[:5]],
            [("req_ready", "i1"), ("resp_valid", "i1"), ("C_addr", "i32"), ("C_wdata", "i32"), ("C_we", "i1")],
        )
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("sel_mw0_addr_n", "state_q == 2 ? SRC_I : ZERO"), assigns)
        self.assertIn(("sel_mw0_wdata_n", "state_q == 2 ? SRC_X : ZERO"), assigns)
        self.assertIn(("sel_mw0_we_n", "state_q == 2 ? TRUE : FALSE"), assigns)
        self.assertIn(("C_addr", "mw0_addr_q"), assigns)
        self.assertIn(("C_wdata", "mw0_wdata_q"), assigns)
        self.assertIn(("C_we", "mw0_we_q"), assigns)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertNotIn(("mw0", "addr", "mw0_addr_n"), attachments)
        self.assertNotIn(("mw0", "wdata", "mw0_wdata_n"), attachments)
        self.assertNotIn(("mw0", "we", "mw0_we_n"), attachments)
        addr_mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_mw0_addr_n")
        wdata_mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_mw0_wdata_n")
        we_mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_mw0_we_n")
        self.assertEqual([(case.key, case.source) for case in addr_mux.cases], [("ZERO", "const_i32_0_n"), ("SRC_I", "i")])
        self.assertEqual([(case.key, case.source) for case in wdata_mux.cases], [("ZERO", "const_i32_0_n"), ("SRC_X", "x")])
        self.assertEqual([(case.key, case.source) for case in we_mux.cases], [("FALSE", "const_i1_0_n"), ("TRUE", "const_i1_1_n")])
        rendered = format_uglir(uglir_design)
        self.assertIn("output C_addr : i32", rendered)
        self.assertIn("output C_wdata : i32", rendered)
        self.assertIn("output C_we : i1", rendered)
        self.assertIn("assign C_addr = mw0_addr_q", rendered)
        self.assertIn("assign C_wdata = mw0_wdata_q", rendered)
        self.assertIn("assign C_we = mw0_we_q", rendered)

    def test_lower_fsm_to_uglir_honors_component_tied_input_ports(self) -> None:
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
                    "bist_en": { "dir": "input", "type": "i1", "tie": "false" },
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

        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("mr0_bist_en_n", "false"), assigns)
        self.assertNotIn(("sel_mr0_bist_en_n", "FALSE"), assigns)
        self.assertNotIn("mx_mr0_bist_en_n", [mux.name for mux in uglir_design.muxes])

    def test_lower_fsm_to_uglir_time_multiplexes_multiple_loads_onto_one_memory_interface(self) -> None:
        fsm_design = parse_uhir(
            """
            design read2
            stage fsm
            schedule kind=control_steps
            input  A : memref<i32>
            input  i : i32
            input  j : i32
            output result : i32
            resources {
              fu mr0 : MEM
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_read2 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state T3 code=16
              state DONE code=32
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> T3
              transition T3 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mr0<-v1]
              emit T1 latch=[r_i32_0]
              emit T2 issue=[mr0<-v2]
              emit T3 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_read2 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = load A, i : i32 class=MEM ii=1 delay=1 start=0 end=0 bind=mr0
              node v2 = load A, j : i32 class=MEM ii=1 delay=1 start=2 end=2 bind=mr0
              node v3 = ret v2 class=CTRL ii=0 delay=0 start=3 end=3
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
              steps [0:3]
              latency 4
              value v1 -> r_i32_0 live=[1:2]
              value v2 -> r_i32_0 live=[3:3]
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

        self.assertEqual([port.name for port in uglir_design.inputs].count("A_rdata"), 1)
        self.assertEqual([port.name for port in uglir_design.outputs].count("A_addr"), 1)
        self.assertEqual(len([resource for resource in uglir_design.resources if resource.kind == "port" and resource.id == "A"]), 1)
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("sel_mr0_addr_n", "state_q == 2 ? SRC_I : state_q == 8 ? SRC_J : ZERO"), assigns)
        self.assertIn(("A_addr", "mr0_addr_q"), assigns)
        addr_mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_mr0_addr_n")
        self.assertEqual([(case.key, case.source) for case in addr_mux.cases], [("ZERO", "const_i32_0_n"), ("SRC_I", "i"), ("SRC_J", "j")])
        rendered = format_uglir(uglir_design)
        self.assertIn("assign A_addr = mr0_addr_q", rendered)

    def test_lower_fsm_to_uglir_resolves_unrolled_issue_occurrences_for_fu_input_muxes(self) -> None:
        fsm_design = parse_uhir(
            """
            design mul_once
            stage fsm
            schedule kind=control_steps
            input  x : i32
            input  y : i32
            output result : i32
            resources {
              fu mul0 : MUL
              reg r_i32_0 : i32
            }
            controller C0 encoding=binary protocol=req_resp completion_order=in_order overlap=true region=proc_mul_once {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=0
              state T0 code=1
              state T1 code=2
              state DONE code=3
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mul0<-v1@0]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_mul_once kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = mul x, y : i32 class=MUL ii=1 delay=2 start=0 end=1 bind=mul0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=2 end=2
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:2]
              latency 3
              value v1 -> r_i32_0 live=[2:2]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "MUL": {
                  "kind": "combinational",
                  "ports": {
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "mul": {
                      "ii": 1,
                      "d": 2,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("sel_mul0_a_n", "state_q == 1 ? SRC_X : ZERO"), assigns)
        self.assertIn(("sel_mul0_b_n", "state_q == 1 ? SRC_Y : ZERO"), assigns)
        mux_a = next(mux for mux in uglir_design.muxes if mux.name == "mx_mul0_a_n")
        mux_b = next(mux for mux in uglir_design.muxes if mux.name == "mx_mul0_b_n")
        self.assertEqual([(case.key, case.source) for case in mux_a.cases], [("ZERO", "const_i32_0_n"), ("SRC_X", "x")])
        self.assertEqual([(case.key, case.source) for case in mux_b.cases], [("ZERO", "const_i32_0_n"), ("SRC_Y", "y")])

    def test_lower_fsm_to_uglir_rejects_same_state_competing_accesses_to_one_memory_interface(self) -> None:
        fsm_design = parse_uhir(
            """
            design read_conflict
            stage fsm
            schedule kind=control_steps
            input  A : memref<i32>
            input  i : i32
            input  j : i32
            output result : i32
            resources {
              fu mr0 : MEM
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_read_conflict {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state DONE code=4
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mr0<-v1, mr0<-v2]
              emit DONE resp_valid=true
            }

            region proc_read_conflict kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = load A, i : i32 class=MEM ii=1 delay=1 start=0 end=0 bind=mr0
              node v2 = load A, j : i32 class=MEM ii=1 delay=1 start=0 end=0 bind=mr0
              node v3 = ret 0:i32 class=CTRL ii=0 delay=0 start=1 end=1
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v0 -> v2
              edge data v1 -> v3
              edge data v2 -> v3
              edge data v3 -> v4
              steps [0:1]
              latency 2
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

        with self.assertRaises(ValueError) as raised:
            lower_fsm_to_uglir(fsm_design, component_library=component_library)

        self.assertIn("one access per memory interface per FSMD state", str(raised.exception))
        self.assertIn("memory 'A'", str(raised.exception))
        self.assertIn("v1, v2", str(raised.exception))

    def test_parse_uglir_roundtrips(self) -> None:
        design = parse_uglir(
            """
            design glue
            stage uglir
            input  clk : clock
            input  rst : i1
            input  req_valid : i1
            input  x : i32
            output req_ready : i1
            output resp_valid : i1
            output result : i32
            resources {
              reg state : u3
              net next_state : u3
              net ewms0_go : i1
              net ewms0_y : i32
              reg r_i32_0 : i32
              net latch_r_i32_0 : i1
              net sel_r_i32_0 : ctrl
              inst ewms0 : EWMS
              mux mx_r_i32_0 : i32
            }

            assign req_ready = state == 0

            ewms0.go(ewms0_go)

            mux mx_r_i32_0 : i32 sel=sel_r_i32_0 {
              hold -> r_i32_0
              ewms0_y -> ewms0_y
            }

            seq clk {
              if rst {
                state <= 0
              } else {
                state <= next_state
                if latch_r_i32_0 {
                  r_i32_0 <= mx_r_i32_0
                }
              }
            }
            """
        )

        self.assertEqual(design.stage, "uglir")
        self.assertEqual(len(design.resources), 9)
        self.assertEqual(design.attachments[0].instance, "ewms0")
        self.assertEqual(design.muxes[0].name, "mx_r_i32_0")
        self.assertEqual(design.seq_blocks[0].updates[1].enable, "latch_r_i32_0")
        rendered = format_uglir(design)
        self.assertIn("stage uglir", rendered)
        self.assertIn("inst ewms0 : EWMS", rendered)
        self.assertIn("ewms0.go(ewms0_go)", rendered)


class UGLIRSyntaxTests(unittest.TestCase):
    """Coverage for textual µglIR parsing and formatting."""

    def test_parse_uglir_with_address_map(self) -> None:
        design = parse_uglir(
            """
            design wrapped
            stage uglir
            input  clk : clock
            input  idx : u2
            output wb_ack_o : i1
            output A_rdata : i32
            const  WB_REG_CONTROL_STATUS = WB_BASE_ADDR + 32'h0000_0000 : u32

            address_map wishbone {
              register control_status offset=32'h0000_0000 access=rw symbol=WB_REG_CONTROL_STATUS
              register x offset=32'h0000_0100 access=rw symbol=WB_REG_IN_X type=i32
              memory A offset=32'h0000_1000 span=32'h0000_0010 access=rw symbol=WB_MEM_A_BASE word_t=i32 depth=4
            }

            resources {
              net wb_req_n : i1
              mem A_mem_q : i32[4]
            }

            assign wb_ack_o = wb_req_n
            assign A_rdata = A_mem_q[idx]

            seq clk {
              A_mem_q[idx] <= 0:i32
            }
            """
        )

        self.assertEqual(len(design.address_maps), 1)
        self.assertEqual(design.address_maps[0].name, "wishbone")
        self.assertEqual(design.address_maps[0].entries[0].kind, "register")
        self.assertEqual(design.address_maps[0].entries[1].attributes["type"], "i32")
        self.assertEqual(design.address_maps[0].entries[2].attributes["depth"], 4)
        self.assertEqual(design.resources[1].kind, "mem")
        self.assertEqual(design.seq_blocks[0].updates[0].target, "A_mem_q[idx]")
        self.assertIn("address_map wishbone {", format_uglir(design))

    def test_parse_uglir_accepts_expression_sequential_enable(self) -> None:
        design = parse_uglir(
            """
            design gated
            stage uglir
            input  clk : clock
            input  a : i1
            input  b : i1
            output y : i1
            resources {
              reg r_q : i1
            }

            assign y = r_q

            seq clk {
              r_q <= false
              if a && b {
                r_q <= true
              }
            }
            """
        )

        self.assertEqual(design.seq_blocks[0].updates[1].enable, "a && b")

    def test_parse_uglir_accepts_multiline_parenthesized_expressions(self) -> None:
        design = parse_uglir(
            """
            design wrapped
            stage uglir
            input  clk : clock
            input  a : i1
            input  b : i1
            output y : i1
            resources {
              reg r_q : i1
              net n_n : i1
            }

            assign n_n = (
              a &&
              b
            )

            assign y = (
              n_n
            )

            seq clk {
              r_q <= false
              if (
                a &&
                b
              ) {
                r_q <= (
                  n_n
                )
              }
            }
            """
        )

        self.assertEqual(design.assigns[0].expr, "a && b")
        self.assertEqual(design.assigns[1].expr, "n_n")
        self.assertEqual(design.seq_blocks[0].updates[1].enable, "a && b")
        self.assertEqual(design.seq_blocks[0].updates[1].value, "n_n")

    def test_format_uglir_wraps_long_expressions_in_parentheses(self) -> None:
        design = parse_uglir(
            """
            design wrapped
            stage uglir
            input  clk : clock
            input  req_valid : i1
            input  req_ready : i1
            output y : i1
            resources {
              reg r_q : i1
              net n_n : i1
            }

            assign n_n = req_valid && req_ready && req_valid && req_ready && req_valid && req_ready && req_valid && req_ready && req_valid && req_ready
            assign y = n_n

            seq clk {
              r_q <= req_valid && req_ready && req_valid && req_ready && req_valid && req_ready && req_valid && req_ready && req_valid && req_ready
            }
            """
        )

        rendered = format_uglir(design)
        self.assertIn("assign n_n = (", rendered)
        self.assertIn("r_q <= (", rendered)
