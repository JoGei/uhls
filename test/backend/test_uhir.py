from __future__ import annotations

import unittest

from uhls.backend.hls.uhir import UHIRParseError, format_uhir, parse_timing_expr, parse_uhir, to_dot


class UHIRParserTests(unittest.TestCase):
    """Coverage for textual µhIR parsing infrastructure."""

    def test_parse_seq_uhir_with_hierarchy_edges_and_maps(self) -> None:
        design = parse_uhir(
            """
            design branch_example
            stage seq

            input  A : memref<i32>
            output C : memref<i32>
            const  N = 16 : i32

            region R0 kind=procedure {
              region_ref R1
              region_ref R2
              region_ref R3

              node c0 = cmp_lt x, y : i1 class=cmp delay=1
              node br0 = branch c0 class=ctrl delay=0

              edge control c0 -> br0
              edge seq br0 -> R1 when=true
              edge seq br0 -> R2 when=false
              edge seq R1 -> R3
              edge seq R2 -> R3
            }

            region R1 kind=then parent=R0 {
              node v1 = add a, b : i32 class=add delay=1
              map v1 <- %t1
            }

            region R2 kind=else parent=R0 {
              node v2 = sub a, b : i32 class=sub delay=1
            }

            region R3 kind=merge parent=R0 {
              node v3 = merge v1, v2 : i32 class=merge delay=0
              node v4 = mul v3, 2 : i32 class=mul delay=2

              edge data v3 -> v4
            }
            """
        )

        self.assertEqual(design.name, "branch_example")
        self.assertEqual(design.stage, "seq")
        self.assertEqual(design.inputs[0].name, "A")
        self.assertEqual(design.outputs[0].name, "C")
        self.assertEqual(design.constants[0].value, 16)
        self.assertIsNone(design.schedule)
        self.assertEqual([region.id for region in design.regions], ["R0", "R1", "R2", "R3"])

        root = design.get_region("R0")
        self.assertIsNotNone(root)
        assert root is not None
        self.assertEqual([ref.target for ref in root.region_refs], ["R1", "R2", "R3"])
        self.assertEqual(root.nodes[0].opcode, "cmp_lt")
        self.assertEqual(root.nodes[1].opcode, "branch")
        self.assertEqual(root.edges[1].attributes["when"], True)
        self.assertEqual(root.edges[2].attributes["when"], False)

        merge = design.get_region("R3")
        self.assertIsNotNone(merge)
        assert merge is not None
        self.assertEqual(merge.nodes[0].operands, ("v1", "v2"))
        self.assertEqual(merge.nodes[1].attributes["delay"], 2)

    def test_parse_sched_uhir_with_region_summary(self) -> None:
        design = parse_uhir(
            """
            design fir
            stage sched
            schedule kind=control_steps

            region R0 kind=procedure {
              node v1 = load A[i] : i32 class=memrd delay=1 start=0 end=0
              node v2 = load B[i] : i32 class=memrd delay=1 start=0 end=0
              node v3 = mul v1, v2 : i32 class=mul delay=2 start=1 end=2
              node v4 = add acc, v3 : i32 class=add delay=1 start=3 end=3 guard=true_path
              node v5 = store C[i], v4 class=memwr delay=1 start=4 end=4

              edge data v1 -> v3
              edge data v2 -> v3
              edge data v3 -> v4
              edge data v4 -> v5

              steps [0:4]
              latency 5
              ii 1
            }
            """
        )

        self.assertEqual(design.stage, "sched")
        self.assertIsNotNone(design.schedule)
        assert design.schedule is not None
        self.assertEqual(design.schedule.kind, "control_steps")
        self.assertEqual(len(design.regions), 1)
        region = design.regions[0]
        self.assertEqual(region.steps, (0, 4))
        self.assertEqual(region.latency, 5)
        self.assertEqual(region.initiation_interval, 1)
        self.assertEqual(region.nodes[2].attributes["start"], 1)
        self.assertEqual(region.nodes[2].attributes["end"], 2)
        self.assertEqual(region.nodes[3].attributes["guard"], "true_path")

    def test_parse_sched_uhir_accepts_symbolic_timing(self) -> None:
        design = parse_uhir(
            """
            design dyn
            stage sched
            schedule kind=hierarchical

            region R0 kind=procedure {
              node v0 = loop child=R1 class=CTRL ii=0 delay=T*ii + rd start=0 end=T*ii + rd - 1
              steps [0:T*ii + rd - 1]
              latency T*ii + rd
              ii ii_loop
            }

            region R1 kind=loop parent=R0 {
              node h0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node h1 = branch c true_child=R2 false_child=R3 class=CTRL ii=0 delay=max(body, exit) start=cmp_t end=cmp_t + max(body, exit) - 1
              node h2 = nop role=sink class=CTRL ii=0 delay=0 start=done_t end=done_t
              edge data h0 -> h1
              edge data h1 -> h2
              steps [0:done_t]
              latency iter_lat
              ii iter_ii
            }

            region R2 kind=body parent=R1 {
              node b0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node b1 = add x, y : i32 class=EWMS ii=1 delay=1 start=1 end=1
              node b2 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data b0 -> b1
              edge data b1 -> b2
            }

            region R3 kind=empty parent=R1 {
              node e0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node e1 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
              edge data e0 -> e1
            }
            """
        )

        self.assertEqual(str(design.regions[0].steps[0]), "0")
        self.assertEqual(str(design.regions[0].steps[1]), "T * ii + rd - 1")
        self.assertEqual(str(design.regions[0].latency), "T * ii + rd")
        self.assertEqual(str(design.regions[0].initiation_interval), "ii_loop")
        self.assertEqual(str(design.regions[0].nodes[0].attributes["delay"]), "T * ii + rd")
        self.assertEqual(str(design.regions[0].nodes[0].attributes["end"]), "T * ii + rd - 1")

    def test_parse_and_format_uhir_preserve_component_library_provenance(self) -> None:
        design = parse_uhir(
            """
            design add1
            stage seq
            component_library "../lib/gen.uhlslib.json"
            component_library "/abs/vendor.uhlslib.json"

            region proc_add1 kind=procedure {
              node v0 = nop role=source
            }
            """
        )

        self.assertEqual(
            design.component_libraries,
            ["../lib/gen.uhlslib.json", "/abs/vendor.uhlslib.json"],
        )
        rendered = format_uhir(design)
        self.assertIn('component_library "../lib/gen.uhlslib.json"', rendered)
        self.assertIn('component_library "/abs/vendor.uhlslib.json"', rendered)
        reparsed = parse_uhir(rendered)
        self.assertEqual(reparsed.component_libraries, design.component_libraries)

    def test_parse_fsm_uhir_with_controller_shell(self) -> None:
        design = parse_uhir(
            """
            design add1
            stage fsm
            schedule kind=control_steps
            resources {
              fu ewms0 : EWMS
              reg r_i32_0 : i32
            }
            controller C0 encoding=binary protocol=req_resp completion_order=in_order overlap=true {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=0
              state T0 code=1
              state DONE code=2
              transition IDLE -> T0 when=req_valid&&req_ready
              transition T0 -> DONE
              transition DONE -> IDLE when=resp_valid&&resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[ewms0<-v1]
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

        self.assertEqual(design.stage, "fsm")
        self.assertEqual(len(design.controllers), 1)
        controller = design.controllers[0]
        self.assertEqual(controller.name, "C0")
        self.assertEqual(controller.attributes["encoding"], "binary")
        self.assertEqual([state.name for state in controller.states], ["IDLE", "T0", "DONE"])
        self.assertEqual(controller.transitions[0].source, "IDLE")
        self.assertEqual(controller.emits[1].attributes["issue"], ("ewms0<-v1",))
        rendered = format_uhir(design)
        self.assertIn("stage fsm", rendered)
        self.assertIn("controller C0 encoding=binary protocol=req_resp completion_order=in_order overlap=true {", rendered)
        self.assertIn("state IDLE code=0", rendered)
        self.assertIn("emit T0 issue=[ewms0<-v1]", rendered)

    def test_parse_fsm_emit_preserves_issue_and_latch_lists(self) -> None:
        design = parse_uhir(
            """
            design add1
            stage fsm
            schedule kind=control_steps
            resources {
              fu ewms0 : EWMS
              reg r0 : i32
              reg r1 : i32
            }
            controller C0 encoding=binary protocol=req_resp completion_order=in_order overlap=true {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=0
              state T0 code=1
              state DONE code=2
              transition IDLE -> T0 when=req_valid&&req_ready
              transition T0 -> DONE
              transition DONE -> IDLE when=resp_valid&&resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[ewms0<-v1, ewms0<-v2] latch=[r0, r1]
              emit DONE resp_valid=true
            }

            region proc_add1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, 1:i32 : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=ewms0
              node v2 = add x, 2:i32 : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=ewms0
              node v3 = ret v2 class=CTRL ii=0 delay=0 start=1 end=1
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v0 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
              steps [0:1]
              latency 2
              value v1 -> r0 live=[1:1]
              value v2 -> r1 live=[1:1]
            }
            """
        )

        emit = design.controllers[0].emits[1]
        self.assertEqual(emit.attributes["issue"], ("ewms0<-v1", "ewms0<-v2"))
        self.assertEqual(emit.attributes["latch"], ("r0", "r1"))

    def test_parse_bind_uhir_accepts_symbolic_fu_only_compat_timing(self) -> None:
        design = parse_uhir(
            """
            design dyn_bind
            stage bind
            schedule kind=hierarchical
            resources {
              fu fu_add0 : FU_ADD
            }

            region proc_dyn_bind kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, y : i32 class=FU_ADD ii=1 delay=1 start=max(0, symb_delay_v4) end=max(0, symb_delay_v4) + 1 - 1 bind=fu_add0
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
              edge data v0 -> v1
              edge data v1 -> v2
              latency max(1, symb_delay_v4)
            }
            """
        )

        self.assertEqual(design.stage, "bind")
        self.assertEqual(str(design.regions[0].nodes[1].attributes["start"]), "symb_delay_v4")

    def test_parse_timing_expr_simplifies_safe_timing_identities(self) -> None:
        self.assertEqual(str(parse_timing_expr("max(0, symb_delay_v4 - 1 + 1)")), "symb_delay_v4")
        self.assertEqual(str(parse_timing_expr("max(0, max(0, symb_delay_v4) + 1 - 1)")), "symb_delay_v4")
        self.assertEqual(
            str(parse_timing_expr("max(symb_delay_v4, symb_delay_v4 + 1, symb_delay_v4 + 2)")),
            "symb_delay_v4 + 2",
        )

    def test_parse_alloc_uhir_requires_class_and_delay_without_schedule(self) -> None:
        design = parse_uhir(
            """
            design fir
            stage alloc

            region R0 kind=procedure {
              node v1 = add a, b : i32 class=alu ii=1 delay=1
              node v2 = ret v1 class=ctrl ii=0 delay=0
              edge data v1 -> v2
            }
            """
        )

        self.assertEqual(design.stage, "alloc")
        self.assertIsNone(design.schedule)
        self.assertEqual(design.regions[0].nodes[0].attributes["class"], "alu")
        self.assertEqual(design.regions[0].nodes[0].attributes["ii"], 1)

    def test_parse_alloc_uhir_accepts_embedded_executability_region(self) -> None:
        design = parse_uhir(
            """
            design fir
            stage alloc

            region R0 kind=procedure {
              node v1 = add a, b : i32 class=alu ii=1 delay=1
              node v2 = ret v1 class=ctrl ii=0 delay=0
              edge data v1 -> v2
            }

            region executability_graph kind=executability {
              node EWMS = fu partition=fu
              node CTRL = fu partition=fu
              node add = op partition=op
              node ret = op partition=op
              edge exg EWMS -- add ii=1 d=1
              edge exg CTRL -- ret ii=0 d=0
            }
            """
        )

        self.assertEqual(design.stage, "alloc")
        exg_region = design.get_region("executability_graph")
        self.assertIsNotNone(exg_region)
        assert exg_region is not None
        self.assertEqual(exg_region.kind, "executability")
        self.assertFalse(exg_region.edges[0].directed)

    def test_parse_seq_uhir_accepts_predicated_select_and_phi_incoming(self) -> None:
        design = parse_uhir(
            """
            design pred
            stage seq

            region R0 kind=procedure {
              node v0 = nop role=source
              node v1 = branch c true_child=R1 false_child=R2 true_label=then false_label=else true_input_label=then false_input_label=else
              node v2 = phi a, b : i32 incoming=[then, else]
              node v3 = add x, y : i32 pred=c
              node v4 = sel c, a, b : i32
              node v5 = nop role=sink
              edge data v1 -> v2
              edge data v4 -> v5
            }

            region R1 kind=basicblock parent=R0 {
              node s = nop role=source
              node t = nop role=sink
              edge data s -> t
            }

            region R2 kind=basicblock parent=R0 {
              node u = nop role=source
              node w = nop role=sink
              edge data u -> w
            }
            """
        )

        region = design.get_region("R0")
        self.assertIsNotNone(region)
        assert region is not None
        phi = next(node for node in region.nodes if node.id == "v2")
        add = next(node for node in region.nodes if node.id == "v3")
        select = next(node for node in region.nodes if node.id == "v4")
        self.assertEqual(phi.attributes["incoming"], ("then", "else"))
        self.assertEqual(add.attributes["pred"], "c")
        self.assertEqual(select.operands, ("c", "a", "b"))

    def test_dot_renders_sel_as_structural_control_node(self) -> None:
        design = parse_uhir(
            """
            design pred
            stage seq

            region R0 kind=procedure {
              node v0 = nop role=source
              node v1 = sel c, a, b : i32
              node v2 = nop role=sink
              edge data v0 -> v1
              edge data v1 -> v2
            }
            """
        )

        dot = to_dot(design)
        self.assertIn('"v1" [label="v1: sel c, a, b : i32", shape=box, style=filled, fillcolor="#e6e6e6"];', dot)

    def test_parse_alloc_uhir_rejects_embedded_executability_missing_used_canonical_op(self) -> None:
        with self.assertRaises(UHIRParseError) as raised:
            parse_uhir(
                """
                design fir
                stage alloc

                region R0 kind=procedure {
                  node v1 = add a, b : i32 class=ALU ii=1 delay=1
                  node v2 = ret v1 class=CTRL ii=0 delay=0
                  edge data v1 -> v2
                }

                region executability_graph kind=executability {
                  node EWMS = fu partition=fu
                  node sub = op partition=op
                  edge exg EWMS -- sub ii=1 d=1
                }
                """
            )

        self.assertIn("does not cover canonical µIR operations used in alloc µhIR", str(raised.exception))

    def test_parse_exg_uhir_with_undirected_weighted_edges(self) -> None:
        design = parse_uhir(
            """
            design fir_exg
            stage exg

            region G0 kind=executability {
              node EWMS = fu partition=fu
              node add = op partition=op
              edge exg EWMS -- add ii=1 d=1
            }
            """
        )

        self.assertEqual(design.stage, "exg")
        self.assertEqual(design.regions[0].nodes[0].opcode, "fu")
        self.assertEqual(design.regions[0].nodes[1].opcode, "op")
        self.assertFalse(design.regions[0].edges[0].directed)
        self.assertEqual(design.regions[0].edges[0].attributes["ii"], 1)
        self.assertEqual(design.regions[0].edges[0].attributes["d"], 1)

    def test_parse_exg_uhir_rejects_non_capitalized_fu_vertex_names(self) -> None:
        with self.assertRaises(UHIRParseError) as raised:
            parse_uhir(
                """
                design bad_exg
                stage exg

                region G0 kind=executability {
                  node ewms = fu partition=fu
                  node add = op partition=op
                  edge exg ewms -- add ii=1 d=1
                }
                """
            )

        self.assertIn("CAPITALIZED", str(raised.exception))

    def test_parse_exg_uhir_rejects_same_partition_edges(self) -> None:
        with self.assertRaises(UHIRParseError) as raised:
            parse_uhir(
                """
                design bad_exg
                stage exg

                region G0 kind=executability {
                  node FU0 = fu partition=fu
                  node FU1 = fu partition=fu
                  edge exg FU0 -- FU1 ii=1 d=1
                }
                """
            )

        self.assertIn("must connect one FU and one op", str(raised.exception))

    def test_parse_uhir_accepts_undirected_edge_syntax(self) -> None:
        design = parse_uhir(
            """
            design rel
            stage seq

            region R0 kind=procedure {
              node a = nop role=source
              node b = nop role=sink
              edge rel a -- b
            }
            """
        )

        edge = design.regions[0].edges[0]
        self.assertFalse(edge.directed)

    def test_parse_exg_uhir_requires_partitioned_undirected_weighted_edges(self) -> None:
        with self.assertRaises(UHIRParseError) as raised:
            parse_uhir(
                """
                design bad_exg
                stage exg

                region G0 kind=executability {
                  node EWMS = fu partition=fu
                  node add = op partition=op
                  edge exg EWMS -> add ii=1 d=1
                }
                """
            )

        self.assertIn("must be undirected", str(raised.exception))

    def test_parse_bind_uhir_with_resources_values_and_muxes(self) -> None:
        design = parse_uhir(
            """
            design fir
            stage bind
            schedule kind=control_steps

            resources {
              fu mul0 : mul
              fu add0 : add
              reg r_acc : i32
              reg r_p : i32
              port mr0 : memrd<word_t=i32,word_len=32> A
              port mw0 : memwr<word_t=i32,word_len=32> C
            }

            region R0 kind=procedure {
              node v1 = load A[i] : i32 class=memrd delay=1 start=0 end=0 bind=mr0
              node v2 = mul v1, 2 : i32 class=mul delay=2 start=1 end=2 bind=mul0
              node v3 = add acc, v2 : i32 class=add delay=1 start=3 end=3 bind=add0
              node v4 = store C[i], v3 class=memwr delay=1 start=4 end=4 bind=mw0

              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4

              map v1 <- %p
              map v3 <- %acc_next

              value %p -> r_p live=[0:1]
              value %acc_next -> r_acc live=[3:4]

              mux mx0 : input=[r_acc, r_next] output=r_acc sel=state
            }
            """
        )

        self.assertEqual(design.stage, "bind")
        self.assertEqual(
            [(resource.kind, resource.id, resource.value, resource.target) for resource in design.resources if resource.kind == "port"],
            [
                ("port", "mr0", "memrd<word_t=i32,word_len=32>", "A"),
                ("port", "mw0", "memwr<word_t=i32,word_len=32>", "C"),
            ],
        )
        self.assertEqual([resource.id for resource in design.resources], ["mul0", "add0", "r_acc", "r_p", "mr0", "mw0"])
        region = design.regions[0]
        self.assertEqual(region.nodes[1].attributes["bind"], "mul0")
        self.assertEqual(region.value_bindings[0].producer, "%p")
        self.assertEqual(region.value_bindings[0].register, "r_p")
        self.assertEqual(region.value_bindings[0].live_intervals, ((0, 1),))
        self.assertEqual(region.muxes[0].inputs, ("r_acc", "r_next"))
        self.assertEqual(region.muxes[0].output, "r_acc")
        self.assertEqual(region.muxes[0].select, "state")

    def test_parse_bind_uhir_accepts_multiple_live_intervals(self) -> None:
        design = parse_uhir(
            """
            design split_live
            stage bind
            schedule kind=control_steps

            resources {
              fu ewms0 : EWMS
              reg r_p : i32
            }

            region R0 kind=procedure {
              node v1 = add a, b : i32 class=EWMS delay=1 start=0 end=0 bind=ewms0
              map v1 <- %p
              value %p -> r_p live=[1:1],[3:4]
            }
            """
        )

        region = design.regions[0]
        self.assertEqual(region.value_bindings[0].live_intervals, ((1, 1), (3, 4)))

    def test_parse_bind_uhir_rejects_legacy_live_interval_syntax(self) -> None:
        with self.assertRaisesRegex(UHIRParseError, r"\[start:end\] syntax"):
            parse_uhir(
                """
                design split_live
                stage bind
                schedule kind=control_steps

                resources {
                  fu ewms0 : EWMS
                  reg r_p : i32
                }

                region R0 kind=procedure {
                  node v1 = add a, b : i32 class=EWMS delay=1 start=0 end=0 bind=ewms0
                  map v1 <- %p
                  value %p -> r_p live=1..1
                }
                """
            )

    def test_parse_sched_uhir_rejects_legacy_steps_interval_syntax(self) -> None:
        with self.assertRaisesRegex(UHIRParseError, r"\[start:end\] syntax"):
            parse_uhir(
                """
                design fir
                stage sched
                schedule kind=control_steps

                region R0 kind=procedure {
                  node v1 = load A[i] : i32 class=memrd delay=1 start=0 end=0
                  node v2 = ret v1 class=CTRL delay=0 start=1 end=1
                  edge data v1 -> v2
                  steps 0..1
                  latency 2
                }
                """
            )

    def test_parse_bind_uhir_allows_unbound_ctrl_nodes(self) -> None:
        design = parse_uhir(
            """
            design fir
            stage bind
            schedule kind=control_steps

            resources {
              fu add0 : add
            }

            region R0 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add a, b : i32 class=add delay=1 start=0 end=0 bind=add0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
            }
            """
        )

        self.assertEqual(design.stage, "bind")
        region = design.regions[0]
        self.assertNotIn("bind", region.nodes[0].attributes)
        self.assertEqual(region.nodes[1].attributes["bind"], "add0")

    def test_sched_stage_requires_schedule_declaration(self) -> None:
        with self.assertRaises(UHIRParseError) as raised:
            parse_uhir(
                """
                design fir
                stage sched

                region R0 kind=procedure {
                  node v1 = add a, b : i32 class=add delay=1 start=0 end=0
                }
                """
            )

        self.assertIn("must declare schedule", str(raised.exception))
