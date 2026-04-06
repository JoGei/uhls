from __future__ import annotations

import json
import unittest

from uhls.backend.hls import lower_fsm_to_uglir
from uhls.backend.uhir import format_uhir, parse_uhir


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
        self.assertIn("state", resource_ids)
        self.assertIn("next_state", resource_ids)
        self.assertIn("ewms0", resource_ids)
        self.assertIn("ewms0_go", resource_ids)
        self.assertIn("ewms0_y", resource_ids)
        self.assertIn("mx_r_i32_0", resource_ids)
        self.assertEqual([(assign.target, assign.expr) for assign in uglir_design.assigns[:5]], [
            ("req_fire", "req_valid & req_ready"),
            ("resp_fire", "resp_valid & resp_ready"),
            ("req_ready", "state == 1"),
            ("resp_valid", "state == 16"),
            ("next_state", "(state == 1 && (req_valid && req_ready)) ? 2 : (state == 2 && (true)) ? 4 : (state == 4 && (true)) ? 8 : (state == 8 && (true)) ? 16 : (state == 16 && (resp_valid && resp_ready)) ? 1 : 1"),
        ])
        self.assertIn(("ewms0", "go", "ewms0_go"), [(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments])
        self.assertIn(("ewms0", "y", "ewms0_y"), [(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments])
        mux = uglir_design.glue_muxes[0]
        self.assertEqual(mux.name, "mx_r_i32_0")
        self.assertEqual(mux.select, "sel_r_i32_0")
        self.assertEqual([(case.key, case.source) for case in mux.cases], [("hold", "r_i32_0"), ("ewms0_y", "ewms0_y")])
        seq_block = uglir_design.seq_blocks[0]
        self.assertEqual(seq_block.clock, "clk")
        self.assertEqual(seq_block.reset, "rst")
        self.assertEqual([(update.target, update.value, update.enable) for update in seq_block.reset_updates], [("state", "1", None)])
        self.assertEqual([(update.target, update.value, update.enable) for update in seq_block.updates], [("state", "next_state", None), ("r_i32_0", "mx_r_i32_0", "latch_r_i32_0")])
        rendered = format_uhir(uglir_design)
        self.assertIn("stage uglir", rendered)
        self.assertIn("input  req_valid : i1", rendered)
        self.assertIn("output req_ready : i1", rendered)
        self.assertIn("inst ewms0 : EWMS", rendered)
        self.assertIn("assign req_ready = state == 1", rendered)
        self.assertIn("ewms0.go(ewms0_go)", rendered)
        self.assertIn("mux mx_r_i32_0 : i32 sel=sel_r_i32_0 {", rendered)
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
        self.assertIn("ewms0_a", resource_ids)
        self.assertIn("ewms0_b", resource_ids)
        self.assertIn("ewms0_op", resource_ids)
        self.assertIn("ewms0_y", resource_ids)
        self.assertNotIn("ewms0_go", resource_ids)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertIn(("ewms0", "a", "ewms0_a"), attachments)
        self.assertIn(("ewms0", "b", "ewms0_b"), attachments)
        self.assertIn(("ewms0", "op", "ewms0_op"), attachments)
        self.assertIn(("ewms0", "y", "ewms0_y"), attachments)
        self.assertIn(("ewms0_a", "x"), {(assign.target, assign.expr) for assign in uglir_design.assigns})
        self.assertIn(("ewms0_b", "1:i32"), {(assign.target, assign.expr) for assign in uglir_design.assigns})
        self.assertIn(("ewms0_op", "0"), {(assign.target, assign.expr) for assign in uglir_design.assigns})
        rendered = format_uhir(uglir_design)
        self.assertIn("assign ewms0_a = x", rendered)
        self.assertIn("assign ewms0_b = 1:i32", rendered)
        self.assertIn("ewms0.op(ewms0_op)", rendered)
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
        self.assertIn("seq0_start", resource_ids)
        self.assertIn("seq0_a", resource_ids)
        self.assertIn("seq0_b", resource_ids)
        self.assertIn("seq0_out", resource_ids)
        self.assertNotIn("seq0_go", resource_ids)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertIn(("seq0", "start", "seq0_start"), attachments)
        self.assertIn(("seq0", "out", "seq0_out"), attachments)
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("seq0_start", "state == 2"), assigns)
        self.assertIn(("seq0_a", "x"), assigns)
        self.assertIn(("seq0_b", "1:i32"), assigns)
        mux = next(mux for mux in uglir_design.glue_muxes if mux.name == "mx_r_i32_0")
        self.assertIn(("seq0_out", "seq0_out"), [(case.key, case.source) for case in mux.cases])
        rendered = format_uhir(uglir_design)
        self.assertIn("assign seq0_start = state == 2", rendered)
        self.assertIn("seq0.start(seq0_start)", rendered)
        self.assertIn("seq0.out(seq0_out)", rendered)
        self.assertNotIn("seq0.go(", rendered)

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
        self.assertIn("state", resource_ids)
        self.assertIn("next_state", resource_ids)
        self.assertIn("C_callee_state", resource_ids)
        self.assertIn("C_callee_next_state", resource_ids)
        self.assertIn("C_callee_act_valid", resource_ids)
        self.assertIn("C_callee_done_ready", resource_ids)
        self.assertIn("C_callee_act_ready", resource_ids)
        self.assertIn("C_callee_done_valid", resource_ids)
        self.assertIn("symb_ready_v1", resource_ids)
        self.assertIn("symb_done_v1", resource_ids)
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("C_callee_act_valid", "state == 1"), assigns)
        self.assertIn(("C_callee_done_ready", "resp_ready"), assigns)
        self.assertIn(("symb_ready_v1", "C_callee_act_ready"), assigns)
        self.assertIn(("symb_done_v1", "C_callee_done_valid"), assigns)
        self.assertIn(("C_callee_act_ready", "C_callee_state == 0"), assigns)
        self.assertIn(("C_callee_done_valid", "C_callee_state == 2"), assigns)
        rendered = format_uhir(uglir_design)
        self.assertIn("reg C_callee_state : u2", rendered)
        self.assertIn("net C_callee_act_valid : i1", rendered)
        self.assertIn("assign symb_done_v1 = C_callee_done_valid", rendered)

    def test_lower_fsm_to_uglir_expands_memref_interface_for_memory_component(self) -> None:
        fsm_design = parse_uhir(
            """
            design read1
            stage fsm
            schedule kind=control_steps
            input  A : memref<i32>
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
        self.assertIn(("port", "A", "MEM", "A"), [(resource.kind, resource.id, resource.value, resource.target) for resource in uglir_design.resources])
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("mr0_addr", "i"), assigns)
        self.assertIn(("A_addr", "mr0_addr"), assigns)
        self.assertIn(("mr0_rdata", "A_rdata"), assigns)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertIn(("mr0", "addr", "mr0_addr"), attachments)
        self.assertIn(("mr0", "rdata", "mr0_rdata"), attachments)
        rendered = format_uhir(uglir_design)
        self.assertIn("input  A_rdata : i32", rendered)
        self.assertIn("output A_addr : i32", rendered)
        self.assertIn("port A : MEM A", rendered)
        self.assertIn("assign mr0_rdata = A_rdata", rendered)
        self.assertIn("assign A_addr = mr0_addr", rendered)

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
        self.assertIn(("mw0_addr", "i"), assigns)
        self.assertIn(("mw0_wdata", "x"), assigns)
        self.assertIn(("mw0_we", "true"), assigns)
        self.assertIn(("C_addr", "mw0_addr"), assigns)
        self.assertIn(("C_wdata", "mw0_wdata"), assigns)
        self.assertIn(("C_we", "mw0_we"), assigns)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertIn(("mw0", "addr", "mw0_addr"), attachments)
        self.assertIn(("mw0", "wdata", "mw0_wdata"), attachments)
        self.assertIn(("mw0", "we", "mw0_we"), attachments)
        rendered = format_uhir(uglir_design)
        self.assertIn("output C_addr : i32", rendered)
        self.assertIn("output C_wdata : i32", rendered)
        self.assertIn("output C_we : i1", rendered)
        self.assertIn("assign mw0_addr = i", rendered)
        self.assertIn("assign mw0_wdata = x", rendered)
        self.assertIn("assign mw0_we = true", rendered)

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
        self.assertIn(("mr0_addr", "state == 2 ? i : state == 8 ? j : 0:i32"), assigns)
        self.assertIn(("A_addr", "mr0_addr"), assigns)
        rendered = format_uhir(uglir_design)
        self.assertIn("assign mr0_addr = state == 2 ? i : state == 8 ? j : 0:i32", rendered)

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
        design = parse_uhir(
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
        self.assertEqual(design.glue_muxes[0].name, "mx_r_i32_0")
        self.assertEqual(design.seq_blocks[0].updates[1].enable, "latch_r_i32_0")
        rendered = format_uhir(design)
        self.assertIn("stage uglir", rendered)
        self.assertIn("inst ewms0 : EWMS", rendered)
        self.assertIn("ewms0.go(ewms0_go)", rendered)
