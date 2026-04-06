from __future__ import annotations

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

