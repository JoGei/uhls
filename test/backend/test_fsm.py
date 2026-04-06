from __future__ import annotations

import unittest

from uhls.backend.hls import CompatibilityBinder, fsm_to_dot, lower_bind_to_fsm, lower_sched_to_bind
from uhls.backend.uhir import parse_uhir


class FSMLoweringTests(unittest.TestCase):
    """Coverage for bind-to-fsm lowering."""

    def test_lower_bind_to_fsm_emits_controller_shell(self) -> None:
        bind_design = parse_uhir(
            """
            design add1
            stage bind
            schedule kind=control_steps
            resources {
              fu ewms0 : EWMS
              reg r_i32_0 : i32
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

        fsm_design = lower_bind_to_fsm(bind_design, encoding="one_hot")

        self.assertEqual(fsm_design.stage, "fsm")
        self.assertEqual(len(fsm_design.controllers), 1)
        controller = fsm_design.controllers[0]
        self.assertEqual(controller.name, "C0")
        self.assertEqual(controller.attributes["encoding"], "one_hot")
        self.assertEqual(controller.attributes["protocol"], "req_resp")
        self.assertEqual(controller.attributes["completion_order"], "in_order")
        self.assertEqual(controller.attributes["overlap"], True)
        self.assertEqual([(port.name, port.type) for port in controller.inputs], [("req_valid", "i1"), ("resp_ready", "i1")])
        self.assertEqual([(port.name, port.type) for port in controller.outputs], [("req_ready", "i1"), ("resp_valid", "i1")])
        self.assertEqual([state.name for state in controller.states], ["IDLE", "T0", "T1", "T2", "DONE"])
        self.assertEqual([state.attributes["code"] for state in controller.states], [1, 2, 4, 8, 16])
        self.assertEqual(
            [(transition.source, transition.target, transition.attributes.get("when")) for transition in controller.transitions],
            [
                ("IDLE", "T0", "req_valid && req_ready"),
                ("T0", "T1", None),
                ("T1", "T2", None),
                ("T2", "DONE", None),
                ("DONE", "IDLE", "resp_valid && resp_ready"),
            ],
        )
        emit_by_state = {emit.state: emit.attributes for emit in controller.emits}
        self.assertEqual(emit_by_state["IDLE"]["req_ready"], True)
        self.assertEqual(emit_by_state["DONE"]["resp_valid"], True)
        self.assertEqual(emit_by_state["T0"]["issue"], ("ewms0<-v1",))
        self.assertEqual(emit_by_state["T1"]["latch"], ("r_i32_0",))
        self.assertEqual([resource.id for resource in fsm_design.resources], ["ewms0", "r_i32_0"])
        region = fsm_design.get_region("proc_add1")
        assert region is not None
        self.assertEqual(region.value_bindings[0].register, "r_i32_0")

    def test_fsm_to_dot_uses_vertical_state_progression(self) -> None:
        bind_design = parse_uhir(
            """
            design add1
            stage bind
            schedule kind=control_steps
            resources {
              fu ewms0 : EWMS
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
            }
            """
        )

        dot = fsm_to_dot(lower_bind_to_fsm(bind_design))
        self.assertIn("rankdir=TB;", dot)

    def test_lower_bind_to_fsm_emits_mux_select_actions(self) -> None:
        bind_design = parse_uhir(
            """
            design acc
            stage bind
            schedule kind=control_steps
            resources {
              fu ewms0 : EWMS
              reg r_acc : i32
            }

            region proc_acc kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add acc, x : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=ewms0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3

              steps [0:1]
              latency 2
              value v1 -> r_acc live=[1:1]
              mux mx0 : input=[r_acc, r_next] output=r_acc sel=state
            }
            """
        )

        fsm_design = lower_bind_to_fsm(bind_design, encoding="binary")

        controller = fsm_design.controllers[0]
        emit_by_state = {emit.state: emit.attributes for emit in controller.emits}
        self.assertEqual(emit_by_state["T1"]["latch"], ("r_acc",))
        self.assertEqual(emit_by_state["T1"]["select"], ("mx0<-state",))

    def test_lower_bind_to_fsm_builds_dynamic_controller_for_fu_only_symbolic_bind(self) -> None:
        bind_design = parse_uhir(
            """
            design dyn
            stage bind
            schedule kind=hierarchical
            resources {
              fu fu_add0 : FU_ADD
            }

            region proc_dyn kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = call child=callee class=CTRL ii=symb_ii_v1 delay=symb_delay_v1 start=0 end=symb_delay_v1 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done
              node v2 = add x, y : i32 class=FU_ADD ii=1 delay=1 start=symb_delay_v1 end=symb_delay_v1 + 1 - 1 bind=fu_add0
              node v3 = ret v2 class=CTRL ii=0 delay=0 start=symb_delay_v1 + 1 end=symb_delay_v1 + 1
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 + 1 end=symb_delay_v1 + 1
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
              edge seq v1 -> callee hierarchy=true
              edge seq callee -> v1 hierarchy=true
            }

            region callee kind=procedure parent=proc_dyn {
              node c0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node c1 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
              edge data c0 -> c1
            }
            """
        )

        fsm_design = lower_bind_to_fsm(bind_design)

        controller = fsm_design.controllers[0]
        self.assertEqual([state.name for state in controller.states], ["IDLE", "P0", "WAIT_v1", "P1", "P2", "DONE"])
        self.assertEqual(
            [(transition.source, transition.target, transition.attributes.get("when")) for transition in controller.transitions],
            [
                ("IDLE", "P0", "req_valid && req_ready && symb_ready_v1"),
                ("P0", "WAIT_v1", None),
                ("WAIT_v1", "P1", "symb_done_v1"),
                ("P1", "P2", None),
                ("P2", "DONE", None),
                ("DONE", "IDLE", "resp_valid && resp_ready"),
            ],
        )
        emit_by_state = {emit.state: emit.attributes for emit in controller.emits}
        self.assertEqual(emit_by_state["IDLE"]["req_ready"], True)
        self.assertEqual(emit_by_state["DONE"]["resp_valid"], True)
        self.assertEqual(emit_by_state["P0"]["activate"], ("v1",))
        self.assertEqual(emit_by_state["P1"]["issue"], ("fu_add0<-v2",))

    def test_lower_bind_to_fsm_allows_concrete_local_register_actions_in_dynamic_child_controller(self) -> None:
        bind_design = parse_uhir(
            """
            design dyn_loop_mux
            stage bind
            schedule kind=hierarchical
            resources {
              fu fu_hdr0 : FU_HDR
              fu fu_body0 : FU_BODY
              reg r_acc : i32
            }

            region proc_dyn_loop kind=procedure {
              region_ref loop_hdr
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = loop child=loop_hdr class=CTRL ii=symb_ii_v1 delay=symb_delay_v1 start=0 end=symb_delay_v1 - 1 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done continue_condition=c iterate_when=c exit_when=!c
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 end=symb_delay_v1
              edge data v0 -> v1
              edge data v1 -> v2
              edge seq v1 -> loop_hdr hierarchy=true
              edge seq loop_hdr -> v1 hierarchy=true
            }

            region loop_hdr kind=loop parent=proc_dyn_loop {
              region_ref loop_body
              region_ref loop_exit
              node h0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node h1 = add i, 1:i32 : i32 class=FU_HDR ii=1 delay=1 start=0 end=0 bind=fu_hdr0
              node h2 = branch c true_child=loop_body false_child=loop_exit class=CTRL ii=0 delay=2 timing=static start=1 end=2
              node h3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data h0 -> h1
              edge data h1 -> h2
              edge data h2 -> h3
              steps [0:2]
              latency 3
            }

            region loop_body kind=body parent=loop_hdr {
              node b0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node b1 = mul x, y : i32 class=FU_BODY ii=1 delay=1 start=1 end=1 bind=fu_body0
              node b2 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data b0 -> b1
              edge data b1 -> b2
              steps [1:2]
              latency 2
              value b1 -> r_acc live=[2:2]
              mux mx_body : input=[r_acc, r_next] output=r_acc sel=body_state
            }

            region loop_exit kind=empty parent=loop_hdr {
              node e0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node e1 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
              edge data e0 -> e1
              steps [1:1]
              latency 0
            }
            """
        )

        fsm_design = lower_bind_to_fsm(bind_design)

        child = next(controller for controller in fsm_design.controllers if controller.name == "C_loop_hdr")
        emit_by_state = {emit.state: emit.attributes for emit in child.emits}
        self.assertEqual(emit_by_state["T1"]["issue"], ("fu_body0<-b1",))
        self.assertEqual(emit_by_state["T2"]["latch"], ("r_acc",))
        self.assertEqual(emit_by_state["T2"]["select"], ("mx_body<-body_state",))

    def test_lower_bind_to_fsm_adds_recursive_child_controller_for_dynamic_loop(self) -> None:
        bind_design = parse_uhir(
            """
            design dyn_loop
            stage bind
            schedule kind=hierarchical
            resources {
              fu fu_hdr0 : FU_HDR
              fu fu_body0 : FU_BODY
            }

            region proc_dyn_loop kind=procedure {
              region_ref loop_hdr
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = loop child=loop_hdr class=CTRL ii=symb_ii_v1 delay=symb_delay_v1 start=0 end=symb_delay_v1 - 1 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done continue_condition=c iterate_when=c exit_when=!c
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 end=symb_delay_v1
              edge data v0 -> v1
              edge data v1 -> v2
              edge seq v1 -> loop_hdr hierarchy=true
              edge seq loop_hdr -> v1 hierarchy=true
            }

            region loop_hdr kind=loop parent=proc_dyn_loop {
              region_ref loop_body
              region_ref loop_exit
              node h0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node h1 = add i, 1:i32 : i32 class=FU_HDR ii=1 delay=1 start=0 end=0 bind=fu_hdr0
              node h2 = branch c true_child=loop_body false_child=loop_exit class=CTRL ii=0 delay=2 timing=static start=1 end=2
              node h3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data h0 -> h1
              edge data h1 -> h2
              edge data h2 -> h3
              steps [0:2]
              latency 3
            }

            region loop_body kind=body parent=loop_hdr {
              node b0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node b1 = mul x, y : i32 class=FU_BODY ii=1 delay=1 start=1 end=1 bind=fu_body0
              node b2 = add b1, z : i32 class=FU_BODY ii=1 delay=1 start=2 end=2 bind=fu_body0
              node b3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data b0 -> b1
              edge data b1 -> b2
              edge data b2 -> b3
              steps [1:2]
              latency 2
            }

            region loop_exit kind=empty parent=loop_hdr {
              node e0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node e1 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
              edge data e0 -> e1
              steps [1:1]
              latency 0
            }
            """
        )

        fsm_design = lower_bind_to_fsm(bind_design)

        self.assertEqual(len(fsm_design.controllers), 2)
        top = fsm_design.controllers[0]
        child = fsm_design.controllers[1]
        self.assertEqual(top.attributes["region"], "proc_dyn_loop")
        self.assertEqual(len(top.links), 1)
        self.assertEqual(top.links[0].child, "C_loop_hdr")
        self.assertEqual(top.links[0].node, "v1")
        self.assertEqual(top.links[0].attributes["act"], ("activate", "act_valid"))
        self.assertEqual(top.links[0].attributes["done"], ("completion", "done_valid"))
        self.assertEqual(child.attributes["protocol"], "act_done")
        self.assertEqual(child.attributes["region"], "loop_hdr")
        self.assertEqual(child.attributes["parent_node"], "v1")
        self.assertEqual([(port.name, port.type) for port in child.inputs], [("act_valid", "i1"), ("done_ready", "i1")])
        self.assertEqual([(port.name, port.type) for port in child.outputs], [("act_ready", "i1"), ("done_valid", "i1")])
        self.assertEqual([state.name for state in child.states], ["IDLE", "T0", "T1", "T2", "DONE"])
        self.assertEqual(
            [(transition.source, transition.target, transition.attributes.get("when")) for transition in child.transitions],
            [
                ("IDLE", "T0", "act_valid && act_ready"),
                ("T0", "T1", "c"),
                ("T0", "DONE", "!c"),
                ("T1", "T2", None),
                ("T2", "T0", None),
                ("DONE", "IDLE", "done_valid && done_ready"),
            ],
        )
        emit_by_state = {emit.state: emit.attributes for emit in child.emits}
        self.assertEqual(emit_by_state["IDLE"]["act_ready"], True)
        self.assertEqual(emit_by_state["DONE"]["done_valid"], True)
        self.assertEqual(emit_by_state["T0"]["issue"], ("fu_hdr0<-h1",))
        self.assertEqual(emit_by_state["T1"]["issue"], ("fu_body0<-b1",))
        self.assertEqual(emit_by_state["T2"]["issue"], ("fu_body0<-b2",))

    def test_lower_bind_to_fsm_adds_recursive_child_controller_for_dynamic_call(self) -> None:
        bind_design = parse_uhir(
            """
            design dyn_call
            stage bind
            schedule kind=hierarchical
            resources {
              fu fu_add0 : FU_ADD
              reg r_acc : i32
            }

            region proc_dyn_call kind=procedure {
              region_ref callee
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = call child=callee class=CTRL ii=symb_ii_v1 delay=symb_delay_v1 start=0 end=symb_delay_v1 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 end=symb_delay_v1
              edge data v0 -> v1
              edge data v1 -> v2
              edge seq v1 -> callee hierarchy=true
              edge seq callee -> v1 hierarchy=true
            }

            region callee kind=procedure parent=proc_dyn_call {
              node c0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node c1 = add x, y : i32 class=FU_ADD ii=1 delay=1 start=0 end=0 bind=fu_add0
              node c2 = ret c1 class=CTRL ii=0 delay=0 start=1 end=1
              node c3 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
              edge data c0 -> c1
              edge data c1 -> c2
              edge data c2 -> c3
              steps [0:1]
              latency 2
              value c1 -> r_acc live=[1:1]
              mux mx_call : input=[r_acc, r_next] output=r_acc sel=call_state
            }
            """
        )

        fsm_design = lower_bind_to_fsm(bind_design)

        self.assertEqual(len(fsm_design.controllers), 2)
        top = fsm_design.controllers[0]
        child = fsm_design.controllers[1]
        self.assertEqual(top.attributes["region"], "proc_dyn_call")
        self.assertEqual(len(top.links), 1)
        self.assertEqual(top.links[0].child, "C_callee")
        self.assertEqual(top.links[0].node, "v1")
        self.assertEqual(child.attributes["protocol"], "act_done")
        self.assertEqual(child.attributes["region"], "callee")
        self.assertEqual(child.attributes["parent_node"], "v1")
        self.assertEqual([(port.name, port.type) for port in child.inputs], [("act_valid", "i1"), ("done_ready", "i1")])
        self.assertEqual([(port.name, port.type) for port in child.outputs], [("act_ready", "i1"), ("done_valid", "i1")])
        self.assertEqual([state.name for state in child.states], ["IDLE", "T0", "T1", "DONE"])
        self.assertEqual(
            [(transition.source, transition.target, transition.attributes.get("when")) for transition in child.transitions],
            [
                ("IDLE", "T0", "act_valid && act_ready"),
                ("T0", "T1", None),
                ("T1", "DONE", None),
                ("DONE", "IDLE", "done_valid && done_ready"),
            ],
        )
        emit_by_state = {emit.state: emit.attributes for emit in child.emits}
        self.assertEqual(emit_by_state["IDLE"]["act_ready"], True)
        self.assertEqual(emit_by_state["DONE"]["done_valid"], True)
        self.assertEqual(emit_by_state["T0"]["issue"], ("fu_add0<-c1",))
        self.assertEqual(emit_by_state["T1"]["latch"], ("r_acc",))
        self.assertEqual(emit_by_state["T1"]["select"], ("mx_call<-call_state",))

    def test_lower_bind_to_fsm_adds_recursive_child_controller_for_dynamic_branch(self) -> None:
        bind_design = parse_uhir(
            """
            design dyn_branch
            stage bind
            schedule kind=hierarchical
            resources {
              fu fu_true0 : FU_TRUE
              fu fu_false0 : FU_FALSE
            }

            region proc_dyn_branch kind=procedure {
              region_ref bb_true
              region_ref bb_false
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = branch c true_child=bb_true false_child=bb_false class=CTRL ii=0 delay=symb_delay_v1 start=0 end=symb_delay_v1 timing=symbolic completion=symb_done_v1 branch_condition=c
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 end=symb_delay_v1
              edge data v0 -> v1
              edge data v1 -> v2
              edge seq v1 -> bb_true hierarchy=true when=true
              edge seq v1 -> bb_false hierarchy=true when=false
              edge seq bb_true -> v1 hierarchy=true when=true
              edge seq bb_false -> v1 hierarchy=true when=false
            }

            region bb_true kind=basic_block parent=proc_dyn_branch {
              node t0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node t1 = add x, y : i32 class=FU_TRUE ii=1 delay=1 start=0 end=0 bind=fu_true0
              node t2 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
              edge data t0 -> t1
              edge data t1 -> t2
              steps [0:1]
              latency 2
            }

            region bb_false kind=basic_block parent=proc_dyn_branch {
              node f0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node f1 = mul a, b : i32 class=FU_FALSE ii=1 delay=1 start=0 end=0 bind=fu_false0
              node f2 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
              edge data f0 -> f1
              edge data f1 -> f2
              steps [0:0]
              latency 1
            }
            """
        )

        fsm_design = lower_bind_to_fsm(bind_design)

        self.assertEqual(len(fsm_design.controllers), 2)
        top = fsm_design.controllers[0]
        child = fsm_design.controllers[1]
        self.assertEqual(top.attributes["region"], "proc_dyn_branch")
        self.assertEqual(len(top.links), 1)
        self.assertEqual(top.links[0].child, "C_v1")
        self.assertEqual(top.links[0].node, "v1")
        self.assertEqual(child.attributes["protocol"], "act_done")
        self.assertEqual(child.attributes["parent_node"], "v1")
        self.assertEqual(child.attributes["region"], "bb_true")
        self.assertEqual(child.attributes["false_region"], "bb_false")
        self.assertEqual(child.attributes["branch_condition"], "c")
        self.assertEqual([(port.name, port.type) for port in child.inputs], [("act_valid", "i1"), ("done_ready", "i1")])
        self.assertEqual([(port.name, port.type) for port in child.outputs], [("act_ready", "i1"), ("done_valid", "i1")])
        self.assertEqual([state.name for state in child.states], ["IDLE", "TRUE_T0", "TRUE_T1", "FALSE_T0", "DONE"])
        self.assertEqual(
            [(transition.source, transition.target, transition.attributes.get("when")) for transition in child.transitions],
            [
                ("IDLE", "TRUE_T0", "act_valid && act_ready && c"),
                ("IDLE", "FALSE_T0", "act_valid && act_ready && !c"),
                ("TRUE_T0", "TRUE_T1", None),
                ("TRUE_T1", "DONE", None),
                ("FALSE_T0", "DONE", None),
                ("DONE", "IDLE", "done_valid && done_ready"),
            ],
        )
        emit_by_state = {emit.state: emit.attributes for emit in child.emits}
        self.assertEqual(emit_by_state["IDLE"]["act_ready"], True)
        self.assertEqual(emit_by_state["DONE"]["done_valid"], True)
        self.assertEqual(emit_by_state["TRUE_T0"]["issue"], ("fu_true0<-t1",))
        self.assertEqual(emit_by_state["FALSE_T0"]["issue"], ("fu_false0<-f1",))

    def test_dynamic_branch_register_bindings_flow_into_recursive_fsm_child(self) -> None:
        sched_design = parse_uhir(
            """
            design dyn_branch_e2e
            stage sched
            schedule kind=hierarchical

            region proc kind=procedure {
              region_ref bb_true
              region_ref bb_false
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = branch c true_child=bb_true false_child=bb_false class=CTRL ii=0 delay=symb_delay_v1 start=0 end=symb_delay_v1 timing=symbolic completion=symb_done_v1 branch_condition=c
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 end=symb_delay_v1
              edge data v0 -> v1
              edge data v1 -> v2
              edge seq v1 -> bb_true hierarchy=true when=true
              edge seq v1 -> bb_false hierarchy=true when=false
              edge seq bb_true -> v1 hierarchy=true when=true
              edge seq bb_false -> v1 hierarchy=true when=false
            }

            region bb_true kind=basic parent=proc {
              node t0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node t1 = add a, b : i32 class=EWMS ii=1 delay=1 start=1 end=1
              node t2 = add t1, k : i32 class=EWMS ii=1 delay=1 start=2 end=2
              node t3 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3
              edge data t0 -> t1
              edge data t1 -> t2
              edge data t2 -> t3
              steps [0:3]
              latency 4
            }

            region bb_false kind=basic parent=proc {
              node f0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node f1 = add c, d : i32 class=EWMS ii=1 delay=1 start=1 end=1
              node f2 = add f1, m : i32 class=EWMS ii=1 delay=1 start=2 end=2
              node f3 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3
              edge data f0 -> f1
              edge data f1 -> f2
              edge data f2 -> f3
              steps [0:3]
              latency 4
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design, binder=CompatibilityBinder())
        self.assertEqual([resource.id for resource in bind_design.resources if resource.kind == "reg"], ["r_i32_0"])

        fsm_design = lower_bind_to_fsm(bind_design)
        child = next(controller for controller in fsm_design.controllers if controller.name == "C_v1")
        emit_by_state = {emit.state: emit.attributes for emit in child.emits}
        self.assertEqual(emit_by_state["TRUE_T1"]["issue"], ("ewms0<-t1",))
        self.assertEqual(emit_by_state["FALSE_T1"]["issue"], ("ewms0<-f1",))
        self.assertEqual(emit_by_state["TRUE_T2"]["latch"], ("r_i32_0",))
        self.assertEqual(emit_by_state["FALSE_T2"]["latch"], ("r_i32_0",))
