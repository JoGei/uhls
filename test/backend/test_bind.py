from __future__ import annotations

import unittest
from pathlib import Path

from uhls.backend.hls import bind_dump_to_dot, binding_to_dot, format_bind_dump, lower_alloc_to_sched, lower_sched_to_bind, parse_bind_dump_spec
from uhls.backend.hls.bind.analysis import _BoundOccurrence, _collapse_template_bound_entries
from uhls.backend.hls.bind.builtin import CompatibilityBinder, LeftEdgeBinder
from uhls.backend.hls.uhir import (
    ExecutabilityGraph,
    create_builtin_gopt_pass,
    lower_module_to_seq,
    lower_seq_to_alloc,
    parse_uhir,
    run_gopt_passes,
)
from uhls.frontend import lower_source_to_uir
from uhls.middleend.passes.opt import CSEPass, ConstPropPass, CopyPropPass, DCEPass, InlineCallsPass, PruneFunctionsPass
from uhls.middleend.passes.util import PassContext, PassManager


class BindingLoweringTests(unittest.TestCase):
    @staticmethod
    def _infer_static(design):
        return run_gopt_passes(
            design,
            [
                create_builtin_gopt_pass("infer_loops"),
                create_builtin_gopt_pass("loop_dialect"),
                create_builtin_gopt_pass("infer_static"),
            ],
        )

    def _static_dot4_relu_sched_design(self):
        source = Path("examples/dot4_relu.c").read_text(encoding="utf-8")
        context = PassContext()
        optimized_module = PassManager(
            [
                InlineCallsPass(),
                PruneFunctionsPass(),
                DCEPass(),
                CSEPass(),
                CopyPropPass(),
                ConstPropPass(),
                DCEPass(),
            ]
        ).run(lower_source_to_uir(source), context)
        seq_design = run_gopt_passes(
            lower_module_to_seq(optimized_module, top="dot4_relu"),
            [
                create_builtin_gopt_pass("infer_loops"),
                create_builtin_gopt_pass("translate_loop_dialect"),
                create_builtin_gopt_pass("infer_static"),
                create_builtin_gopt_pass("simplify_static_control"),
                create_builtin_gopt_pass("predicate"),
                create_builtin_gopt_pass("fold_predicates"),
            ],
        )
        alloc_design = lower_seq_to_alloc(seq_design, executability_graph=self._full_executability_graph())
        return lower_alloc_to_sched(alloc_design)

    def _full_executability_graph(self) -> ExecutabilityGraph:
        operations = (
            "add",
            "sub",
            "mul",
            "div",
            "mod",
            "and",
            "or",
            "xor",
            "shl",
            "shr",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "neg",
            "not",
            "mov",
            "const",
            "load",
            "store",
            "phi",
            "call",
            "print",
            "param",
            "br",
            "cbr",
            "ret",
        )
        return ExecutabilityGraph(
            functional_units=("EWMS",),
            operations=operations,
            edges=tuple(("EWMS", operation, 1, 1) for operation in operations),
        )

    def test_conflict_like_template_collapse_merges_occurrence_suffixes(self) -> None:
        collapsed = _collapse_template_bound_entries(
            [
                _BoundOccurrence("operation", "R0", "R0", "v1@0", "v1@0", "mul", "mul0", "MUL", 0, 1),
                _BoundOccurrence("operation", "R0", "R0", "v1@1", "v1@1", "mul", "mul0", "MUL", 2, 3),
                _BoundOccurrence("register", "R0", "R0", "v2@0", "reg_v2@0_b1", "reg", "r0", "i32", 1, 2),
                _BoundOccurrence("register", "R0", "R0", "v2@1", "reg_v2@1_b1", "reg", "r0", "i32", 3, 4),
            ]
        )

        self.assertEqual(
            [(entry.display_id, entry.start, entry.end) for entry in collapsed],
            [("v1", 0, 3), ("reg_v2_b1", 1, 4)],
        )

    def test_lower_sched_to_bind_reuses_one_resource_for_non_overlapping_ops(self) -> None:
        sched_design = parse_uhir(
            """
            design add_chain
            stage sched
            schedule kind=control_steps

            region proc_add_chain kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add a, b : i32 class=FU_ADD ii=1 delay=1 start=0 end=0
              node v2 = add v1, c : i32 class=FU_ADD ii=1 delay=1 start=1 end=1
              node v3 = ret v2 class=CTRL ii=0 delay=0 start=2 end=2
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4

              steps [0:2]
              latency 3
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design)
        region = bind_design.get_region("proc_add_chain")
        self.assertIsNotNone(region)
        assert region is not None

        add_nodes = [node for node in region.nodes if node.opcode == "add"]
        self.assertEqual(bind_design.stage, "bind")
        self.assertEqual([resource.id for resource in bind_design.resources], ["fu_add0", "r_i32_0"])
        self.assertEqual(add_nodes[0].attributes["bind"], "fu_add0")
        self.assertEqual(add_nodes[1].attributes["bind"], "fu_add0")
        self.assertEqual(
            [(binding.producer, binding.register, binding.live_start, binding.live_end) for binding in region.value_bindings],
            [("v1", "r_i32_0", 1, 1), ("v2", "r_i32_0", 2, 2)],
        )
        self.assertEqual([binding.live_intervals for binding in region.value_bindings], [((1, 1),), ((2, 2),)])
        self.assertNotIn("bind", next(node for node in region.nodes if node.opcode == "ret").attributes)

    def test_lower_sched_to_bind_uses_distinct_resources_for_overlapping_ops(self) -> None:
        sched_design = parse_uhir(
            """
            design add_pair
            stage sched
            schedule kind=control_steps

            region proc_add_pair kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add a, b : i32 class=FU_ADD ii=1 delay=1 start=0 end=0
              node v2 = add c, d : i32 class=FU_ADD ii=1 delay=1 start=0 end=0
              node v3 = mul v1, v2 : i32 class=FU_MUL ii=1 delay=1 start=1 end=1
              node v4 = ret v3 class=CTRL ii=0 delay=0 start=2 end=2
              node v5 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

              edge data v0 -> v1
              edge data v0 -> v2
              edge data v1 -> v3
              edge data v2 -> v3
              edge data v3 -> v4
              edge data v4 -> v5

              steps [0:2]
              latency 3
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design)
        region = bind_design.get_region("proc_add_pair")
        assert region is not None

        add_bindings = sorted(node.attributes["bind"] for node in region.nodes if node.opcode == "add")
        self.assertEqual(add_bindings, ["fu_add0", "fu_add1"])
        self.assertEqual(
            [resource.id for resource in bind_design.resources],
            ["fu_add0", "fu_add1", "fu_mul0", "r_i32_0", "r_i32_1"],
        )
        self.assertEqual(
            [(binding.producer, binding.register, binding.live_start, binding.live_end) for binding in region.value_bindings],
            [("v1", "r_i32_0", 1, 1), ("v2", "r_i32_1", 1, 1), ("v3", "r_i32_0", 2, 2)],
        )

    def test_lower_sched_to_bind_prefers_mapped_value_ids_in_value_bindings(self) -> None:
        sched_design = parse_uhir(
            """
            design mapped_values
            stage sched
            schedule kind=control_steps

            region proc_mapped_values kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add a, b : i32 class=FU_ADD ii=1 delay=1 start=0 end=0
              node v2 = add v1, c : i32 class=FU_ADD ii=1 delay=1 start=1 end=1
              node v3 = ret v2 class=CTRL ii=0 delay=0 start=2 end=2
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4

              map v1 <- t1_0
              map v2 <- t2_0
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design)
        region = bind_design.get_region("proc_mapped_values")
        assert region is not None
        self.assertEqual(
            [(binding.producer, binding.register, binding.live_start, binding.live_end) for binding in region.value_bindings],
            [("t1_0", "r_i32_0", 1, 1), ("t2_0", "r_i32_0", 2, 2)],
        )

    def test_lower_sched_to_bind_preserves_dynamic_call_contract(self) -> None:
        sched_design = parse_uhir(
            """
            design dyn_contract
            stage sched
            schedule kind=hierarchical

            region proc_dyn_contract kind=procedure {
              region_ref proc_callee
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v2 = call x child=proc_callee class=CTRL ii=II delay=symb_delay_v2 start=1 end=symb_delay_v2 timing=symbolic completion=symb_done_v2 ready=symb_ready_v2 handshake=ready_done
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

              edge data v0 -> v2
              edge data v2 -> v3
            }

            region proc_callee kind=procedure {
              node c0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node c1 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
              edge data c0 -> c1
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design)
        region = bind_design.get_region("proc_dyn_contract")
        assert region is not None
        call_node = next(node for node in region.nodes if node.opcode == "call")
        self.assertEqual(str(call_node.attributes["ii"]), "II")
        self.assertEqual(call_node.attributes["ready"], "symb_ready_v2")
        self.assertEqual(call_node.attributes["completion"], "symb_done_v2")
        self.assertEqual(call_node.attributes["handshake"], "ready_done")

    def test_lower_sched_to_bind_infers_memory_port_resource(self) -> None:
        sched_design = parse_uhir(
            """
            design mem_bind
            stage sched
            schedule kind=control_steps
            input  A : memref<i32, 4>
            input  i : i32
            output result : i32

            region proc_mem_bind kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = load A, i : i32 class=MEM ii=1 delay=1 start=0 end=0
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

        bind_design = lower_sched_to_bind(sched_design)

        self.assertIn(
            ("port", "A", "MEM<word_t=i32,word_len=4>", "A"),
            [(resource.kind, resource.id, resource.value, resource.target) for resource in bind_design.resources],
        )

    def test_compat_binder_infers_memory_port_resource(self) -> None:
        sched_design = parse_uhir(
            """
            design dyn_mem_bind
            stage sched
            schedule kind=hierarchical
            input  A : memref<i32>
            input  i : i32
            output result : i32

            region proc_dyn_mem_bind kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = load A, i : i32 class=MEM ii=1 delay=1 start=symb_delay_v0 end=symb_delay_v0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design, binder=CompatibilityBinder())

        self.assertIn(
            ("port", "A", "MEM<word_t=i32>", "A"),
            [(resource.kind, resource.id, resource.value, resource.target) for resource in bind_design.resources],
        )

    def test_lower_sched_to_bind_rejects_symbolic_sched_timing(self) -> None:
        sched_design = parse_uhir(
            """
            design dyn_sched
            stage sched
            schedule kind=hierarchical

            region proc_dyn_sched kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, y : i32 class=FU_ADD ii=1 delay=1 start=max(0, symb_delay_v4) end=max(0, symb_delay_v4) + 1 - 1
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
              edge data v0 -> v1
              edge data v1 -> v2
            }
            """
        )

        with self.assertRaisesRegex(ValueError, "requires concrete sched timing"):
            lower_sched_to_bind(sched_design)

    def test_compat_binder_accepts_symbolic_sched_timing(self) -> None:
        sched_design = parse_uhir(
            """
            design dyn_compat
            stage sched
            schedule kind=hierarchical

            region proc_dyn_compat kind=procedure {
              region_ref proc_callee
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = call x child=proc_callee class=CTRL ii=II delay=symb_delay_v1 start=0 end=symb_delay_v1 - 1 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done
              node v2 = add a, b : i32 class=FU_ADD ii=1 delay=1 start=max(0, symb_delay_v1) end=max(0, symb_delay_v1) + 1 - 1
              node v3 = add v2, c : i32 class=FU_ADD ii=1 delay=1 start=max(0, max(0, symb_delay_v1) + 1) end=max(0, max(0, symb_delay_v1) + 1) + 1 - 1
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0

              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
            }

            region proc_callee kind=procedure {
              node c0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node c1 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
              edge data c0 -> c1
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design, binder=CompatibilityBinder())
        region = bind_design.get_region("proc_dyn_compat")
        assert region is not None
        add_nodes = [node for node in region.nodes if node.opcode == "add"]
        self.assertEqual([resource.id for resource in bind_design.resources], ["fu_add0"])
        self.assertEqual([node.attributes["bind"] for node in add_nodes], ["fu_add0", "fu_add0"])
        self.assertEqual(region.value_bindings, [])

    def test_compat_binder_reuses_fu_across_mutually_exclusive_dynamic_branch_arms(self) -> None:
        sched_design = parse_uhir(
            """
            design dyn_branch_share
            stage sched
            schedule kind=hierarchical

            region proc_dyn_branch_share kind=procedure {
              region_ref bb_true
              region_ref bb_false
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = branch c true_child=bb_true false_child=bb_false class=CTRL ii=0 delay=symb_delay_v1 start=1 end=symb_delay_v1 timing=symbolic completion=symb_done_v1 branch_condition=c
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

              edge data v0 -> v1
              edge data v1 -> v2
            }

            region bb_true kind=basic parent=proc_dyn_branch_share {
              node t0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node t1 = add a, b : i32 class=EWMS ii=1 delay=1 start=2 end=2
              node t2 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

              edge data t0 -> t1
              edge data t1 -> t2
            }

            region bb_false kind=basic parent=proc_dyn_branch_share {
              node f0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node f1 = add c, d : i32 class=EWMS ii=1 delay=1 start=2 end=2
              node f2 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

              edge data f0 -> f1
              edge data f1 -> f2
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design, binder=CompatibilityBinder())
        true_region = bind_design.get_region("bb_true")
        false_region = bind_design.get_region("bb_false")
        assert true_region is not None
        assert false_region is not None

        true_add = next(node for node in true_region.nodes if node.opcode == "add")
        false_add = next(node for node in false_region.nodes if node.opcode == "add")
        self.assertEqual(true_add.attributes["bind"], false_add.attributes["bind"])
        self.assertEqual([resource.id for resource in bind_design.resources if resource.kind == "fu"], ["ewms0"])

    def test_compat_binder_reuses_register_across_mutually_exclusive_dynamic_branch_values(self) -> None:
        sched_design = parse_uhir(
            """
            design dyn_branch_value_share
            stage sched
            schedule kind=hierarchical

            region proc_dyn_branch_value_share kind=procedure {
              region_ref bb_true
              region_ref bb_false
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = branch c true_child=bb_true false_child=bb_false class=CTRL ii=0 delay=symb_delay_v1 start=1 end=symb_delay_v1 timing=symbolic completion=symb_done_v1 branch_condition=c
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

              edge data v0 -> v1
              edge data v1 -> v2
            }

            region bb_true kind=basic parent=proc_dyn_branch_value_share {
              node t0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node t1 = add a, b : i32 class=EWMS ii=1 delay=1 start=2 end=2
              node t2 = add t1, k : i32 class=EWMS ii=1 delay=1 start=3 end=3
              node t3 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4

              edge data t0 -> t1
              edge data t1 -> t2
              edge data t2 -> t3
            }

            region bb_false kind=basic parent=proc_dyn_branch_value_share {
              node f0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node f1 = add c, d : i32 class=EWMS ii=1 delay=1 start=2 end=2
              node f2 = add f1, m : i32 class=EWMS ii=1 delay=1 start=3 end=3
              node f3 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4

              edge data f0 -> f1
              edge data f1 -> f2
              edge data f2 -> f3
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design, binder=CompatibilityBinder())
        true_region = bind_design.get_region("bb_true")
        false_region = bind_design.get_region("bb_false")
        assert true_region is not None
        assert false_region is not None

        self.assertEqual([resource.id for resource in bind_design.resources if resource.kind == "reg"], ["r_i32_0"])
        self.assertEqual([(binding.producer, binding.register) for binding in true_region.value_bindings], [("t1", "r_i32_0")])
        self.assertEqual([(binding.producer, binding.register) for binding in false_region.value_bindings], [("f1", "r_i32_0")])
        self.assertEqual(true_region.value_bindings[0].live_intervals, ((3, 3),))
        self.assertEqual(false_region.value_bindings[0].live_intervals, ((3, 3),))

    def test_format_bind_dump_marks_fu_only_binding_mode(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
            """
            design fu_only_mode
            stage sched
            schedule kind=control_steps

            region proc_fu_only_mode kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, y : i32 class=FU_ADD ii=1 delay=1 start=0 end=0
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
              edge data v0 -> v1
              edge data v1 -> v2
            }
            """
            ),
            binder=CompatibilityBinder(),
        )
        rendered = format_bind_dump(bind_design, ("conflict",))
        self.assertIn("binding_mode fu_only", rendered)

    def test_bind_dump_rejects_symbolic_compat_artifact(self) -> None:
        sched_design = parse_uhir(
            """
            design dyn_compat_dump
            stage sched
            schedule kind=hierarchical

            region proc_dyn_compat_dump kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, y : i32 class=FU_ADD ii=1 delay=1 start=max(0, symb_delay_v4) end=max(0, symb_delay_v4) + 1 - 1
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
              edge data v0 -> v1
              edge data v1 -> v2
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design, binder=CompatibilityBinder())
        with self.assertRaisesRegex(ValueError, "requires concrete bind timing"):
            format_bind_dump(bind_design, ("conflict",))

    def test_lower_sched_to_bind_extends_loop_carried_value_liveness_to_header_phi(self) -> None:
        sched_design = parse_uhir(
            """
            design loop_carried
            stage sched
            schedule kind=hierarchical

            region proc_loop_carried kind=procedure {
              region_ref loop_header_1
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = loop static_trip_count=4 child=loop_header_1 class=CTRL ii=0 delay=20 start=0 end=19 iter_latency=5 iter_initiation_interval=5 iter_ramp_down=0
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=20 end=20

              edge data v0 -> v1
              edge data v1 -> v2
            }

            region loop_header_1 kind=loop parent=proc_loop_carried {
              region_ref loop_body_1
              region_ref loop_exit_1
              node v3 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v4 = phi 0 : i32, t4_0 : i32 class=EWMS ii=1 delay=1 start=0 end=0
              node v5 = branch c true_child=loop_body_1 false_child=loop_exit_1 class=CTRL ii=0 delay=3 start=1 end=3
              node v6 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4

              edge data v3 -> v4
              edge data v4 -> v5
              edge data v5 -> v6
            }

            region loop_body_1 kind=body parent=loop_header_1 {
              node v10 = nop role=source class=CTRL ii=0 delay=0 start=2 end=2
              node v13 = add i_1, 1 : i32 : i32 class=EWMS ii=1 delay=1 start=2 end=2
              node v16 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

              edge data v10 -> v13
              edge data v13 -> v16

              map v13 <- t4_0
            }

            region loop_exit_1 kind=empty parent=loop_header_1 {
              node e0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node e1 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1

              edge data e0 -> e1
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design)
        body_region = bind_design.get_region("loop_body_1")
        assert body_region is not None
        self.assertEqual(
            [(binding.producer, binding.live_start, binding.live_end) for binding in body_region.value_bindings],
            [("t4_0", 3, 5)],
        )

    def test_binding_to_dot_renders_conflicts_and_coloring(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design add_pair
                stage sched
                schedule kind=control_steps

                region proc_add_pair kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = add c, d : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v3 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
                  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

                  edge data v0 -> v1
                  edge data v0 -> v2
                  edge data v1 -> v3
                  edge data v2 -> v3
                  edge data v3 -> v4
                }
                """
            )
        )

        dot = binding_to_dot(bind_design)
        self.assertIn('digraph "add_pair.bind.dump"', dot)
        self.assertIn('subgraph "cluster_conflict_operation_ewms"', dot)
        self.assertIn('label="EWMS operation conflict graph"', dot)
        self.assertIn('subgraph "cluster_conflict_operation_ctrl"', dot)
        self.assertIn('label="CTRL operation conflict graph"', dot)
        self.assertIn('subgraph "cluster_conflict_register_i32"', dot)
        self.assertIn('label="i32 register conflict graph"', dot)
        self.assertIn('"v1" -> "v2" [label="proc_add_pair"', dot)
        self.assertIn('"v1" [label="v1 add ewms0"', dot)
        self.assertIn('"v2" [label="v2 add ewms1"', dot)
        self.assertIn('"reg_v1" [label="val=v1 bind=r_i32_0"', dot)

    def test_binding_to_dot_supports_compact_labels(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design add_pair
                stage sched
                schedule kind=control_steps

                region proc_add_pair kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
                  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                }
                """
            )
        )

        dot = binding_to_dot(bind_design, compact=True)
        self.assertIn('"v1" [label="v1 + ewms0"', dot)

    def test_format_bind_dump_supports_trp_unroll_and_dfgsb(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design add1
                stage sched
                schedule kind=control_steps

                region proc_add1 kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
                  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                }
                """
            )
        )

        rendered = format_bind_dump(bind_design, ("trp", "trp_unroll", "dfgsb", "dfgsb_unroll"))
        self.assertIn("bind_dump trp", rendered)
        self.assertIn("region proc_add1 (time-resource plane)", rendered)
        self.assertIn("ctrl", rendered)
        self.assertIn("ewms0", rendered)
        self.assertIn("bind_dump trp_unroll", rendered)
        self.assertIn("global (time-resource plane unrolled)", rendered)
        self.assertIn("bind_dump dfgsb", rendered)
        self.assertIn("----- proc_add1 kind=procedure -----", rendered)
        self.assertIn("reg 1", rendered)
        self.assertIn("bind_dump dfgsb_unroll", rendered)
        self.assertIn("global (dataflow graph with schedule and binding, unrolled)", rendered)

    def test_bind_dump_to_dot_supports_compatibility_graph(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design add_chain
                stage sched
                schedule kind=control_steps

                region proc_add_chain kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = add v1, c : i32 class=EWMS ii=1 delay=1 start=1 end=1
                  node v3 = ret v2 class=CTRL ii=0 delay=0 start=2 end=2
                  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                  edge data v3 -> v4
                }
                """
            )
        )

        dot = bind_dump_to_dot(bind_design, ("compatibility",))
        self.assertIn('label="EWMS operation compatibility graph"', dot)
        self.assertIn('"v1" -> "v2" [label="proc_add_chain", color="#2ca02c", dir=none]', dot)
        self.assertIn('label="i32 register compatibility graph"', dot)

    def test_parse_bind_dump_spec_accepts_compa_alias(self) -> None:
        self.assertEqual(parse_bind_dump_spec("compa"), ("compatibility",))
        self.assertEqual(parse_bind_dump_spec("conflict,compa"), ("conflict", "compatibility"))

    def test_format_bind_dump_includes_register_conflict_graph(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design add1
                stage sched
                schedule kind=control_steps

                region proc_add1 kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
                  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                }
                """
            )
        )

        rendered = format_bind_dump(bind_design, ("conflict",))
        self.assertIn("class i32 (register conflict graph)", rendered)
        self.assertIn("node v1 opcode=reg bind=r_i32_0 interval=1..1", rendered)

    def test_register_liveness_extends_through_consumer_ii(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design hold_input
                stage sched
                schedule kind=control_steps

                region proc_hold_input kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = mul v1, c : i32 class=EWMS ii=2 delay=2 start=1 end=2
                  node v3 = ret v2 class=CTRL ii=0 delay=0 start=3 end=3
                  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                  edge data v3 -> v4
                }
                """
            )
        )

        region = bind_design.get_region("proc_hold_input")
        assert region is not None
        self.assertEqual(
            [(binding.producer, binding.live_start, binding.live_end) for binding in region.value_bindings],
            [("v1", 1, 2), ("v2", 3, 3)],
        )

    def test_dfgsb_dot_routes_multicycle_operand_into_first_operation_cycle(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design hold_input
                stage sched
                schedule kind=control_steps

                region proc_hold_input kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = mul v1, c : i32 class=EWMS ii=2 delay=2 start=1 end=2
                  node v3 = ret v2 class=CTRL ii=0 delay=0 start=3 end=3
                  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                  edge data v3 -> v4
                }
                """
            )
        )

        dot = bind_dump_to_dot(bind_design, ("dfgsb",))
        self.assertIn('"dfgsb_proc_hold_input_reg_1_2" -> "dfgsb_proc_hold_input_op_1_1" [color="#1f78b4", penwidth=1.3, label="v1"', dot)
        self.assertNotIn('"dfgsb_proc_hold_input_reg_1_2" -> "dfgsb_proc_hold_input_op_2_1" [color="#1f78b4", penwidth=1.3, label="v1"', dot)

    def test_trp_unroll_keeps_post_loop_ret_after_last_iteration_result(self) -> None:
        source = """
        int32_t dot4(int32_t A[4], int32_t B[4]) {
            int32_t i;
            int32_t sum = 0;
            for (i = 0; i < 4; i = i + 1) {
                sum = sum + A[i] * B[i];
            }
            return sum;
        }
        """
        bind_design = lower_sched_to_bind(
            lower_alloc_to_sched(
                lower_seq_to_alloc(
                    self._infer_static(lower_module_to_seq(lower_source_to_uir(source), top="dot4")),
                    executability_graph=self._full_executability_graph(),
                )
            )
        )

        rendered = format_bind_dump(bind_design, ("trp_unroll",))
        lines = rendered.splitlines()
        ret_line = next(line for line in lines if " ret" in line)
        last_add_line = max(line for line in lines if "@3 add" in line)
        time_lines = [
            line
            for line in lines
            if "|" in line and line.split("|", 1)[0].strip().isdigit()
        ]
        last_time_line = max(time_lines, key=lambda line: int(line.split("|", 1)[0].strip()))
        ret_time = int(ret_line.split("|", 1)[0].strip())
        last_add_time = int(last_add_line.split("|", 1)[0].strip())
        self.assertIn("v7 ret", ret_line)
        self.assertIn("@3 add", last_add_line)
        self.assertNotIn(" ret", last_add_line)
        self.assertEqual(int(last_time_line.split("|", 1)[0].strip()), ret_time)
        self.assertGreater(ret_time, last_add_time)

    def test_flattened_binding_colors_register_values_per_occurrence(self) -> None:
        sched_design = self._static_dot4_relu_sched_design()

        bind_design = lower_sched_to_bind(sched_design, binder=LeftEdgeBinder(flatten=True))

        body_region = bind_design.get_region("loop_body_dot4_relu_for_header_1")
        self.assertIsNotNone(body_region)
        assert body_region is not None

        sum_like_bindings = [binding for binding in body_region.value_bindings if binding.producer == "inl_mac_0_t1_0"]
        self.assertEqual(len(sum_like_bindings), 1)
        self.assertGreater(len(sum_like_bindings[0].live_intervals), 1)

    def test_non_flattened_trp_unroll_keeps_iteration_specific_register_values(self) -> None:
        sched_design = self._static_dot4_relu_sched_design()

        bind_design = lower_sched_to_bind(sched_design)

        rendered = format_bind_dump(bind_design, ("trp_unroll",))
        self.assertIn("t1_0@1", rendered)
        self.assertIn("inl_mac_0_t1_0@0_b3", rendered)
        self.assertIn("inl_mac_0_t1_0@2_b3", rendered)
        self.assertNotIn("sum_1[t=1]_b1", rendered)

    def test_non_flattened_dfgsb_unroll_renders_later_iterations_and_hidden_branch_flow(self) -> None:
        sched_design = self._static_dot4_relu_sched_design()

        bind_design = lower_sched_to_bind(sched_design)

        dot = bind_dump_to_dot(bind_design, ("dfgsb_unroll",))
        self.assertIn('"dfgsb_unroll_op_12_4" -> "dfgsb_unroll_reg_13_7" [color="#1f78b4", penwidth=1.3, label="t5_0"', dot)
        self.assertIn('"dfgsb_unroll_reg_13_7" -> "dfgsb_unroll_op_13_2" [color="#1f78b4", penwidth=1.3, label="t5_0"', dot)
        self.assertIn('"dfgsb_unroll_op_13_2" -> "dfgsb_unroll_op_13_3"', dot)

    def test_flattened_dfgsb_unroll_renders_final_loop_value_into_post_loop_consumer(self) -> None:
        sched_design = self._static_dot4_relu_sched_design()

        bind_design = lower_sched_to_bind(sched_design, binder=LeftEdgeBinder(flatten=True))

        dot = bind_dump_to_dot(bind_design, ("dfgsb_unroll",))
        self.assertIn('"dfgsb_unroll_op_11_4" -> "dfgsb_unroll_reg_12_8" [color="#1f78b4", penwidth=1.3, label="inl_mac_0_t1_0"', dot)
        self.assertIn('"dfgsb_unroll_reg_12_8" -> "dfgsb_unroll_op_12_4" [color="#1f78b4", penwidth=1.3, label="inl_mac_0_t1_0"', dot)
        self.assertIn('"dfgsb_unroll_reg_12_8" -> "dfgsb_unroll_op_13_2" [color="#1f78b4", penwidth=1.3, label="inl_mac_0_t1_0"', dot)
        self.assertIn('"dfgsb_unroll_op_13_2" -> "dfgsb_unroll_op_13_3"', dot)

    def test_dfgsb_dot_routes_cross_sgu_value_through_register_without_direct_duplicate_edge(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design child_use
                stage sched
                schedule kind=hierarchical

                region proc_child_use kind=procedure {
                  region_ref bb_child
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = call child=bb_child class=CTRL ii=0 delay=1 start=1 end=1
                  node v3 = ret x class=CTRL ii=0 delay=0 start=3 end=3
                  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                  edge data v3 -> v4
                }

                region bb_child kind=basic parent=proc_child_use {
                  node c0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
                  node c1 = add v1, c : i32 class=EWMS ii=1 delay=1 start=1 end=1
                  node c2 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

                  edge data c0 -> c1
                  edge data c1 -> c2
                }
                """
            )
        )

        proc_region = bind_design.get_region("proc_child_use")
        assert proc_region is not None
        self.assertEqual(
            [(binding.producer, binding.live_intervals) for binding in proc_region.value_bindings],
            [("v1", ((1, 2),))],
        )

        dot = bind_dump_to_dot(bind_design, ("dfgsb",))
        self.assertIn('label="r_i32_0"', dot)
        self.assertIn('"dfgsb_proc_child_use_op_0_2" -> "dfgsb_proc_child_use_reg_1_3" [color="#1f78b4", penwidth=1.3, label="v1"', dot)
        self.assertIn('"dfgsb_proc_child_use_reg_1_3" -> "dfgsb_bb_child_op_1_0" [color="#1f78b4", penwidth=1.1, style=dashed, label="v1"', dot)
        self.assertNotIn('"dfgsb_proc_child_use_reg_2_2"', dot)
        self.assertNotIn('"dfgsb_proc_child_use_op_0_0" -> "dfgsb_bb_child_op_1_0"', dot)

    def test_bind_dump_to_dot_supports_dfgsb_unroll(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design add1
                stage sched
                schedule kind=control_steps

                region proc_add1 kind=procedure {
                  map v1 <- t1_0
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
                  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                }
                """
            )
        )

        dot = bind_dump_to_dot(bind_design, ("dfgsb_unroll",))
        self.assertIn('subgraph "cluster_dfgsb_unroll"', dot)
        self.assertIn('label="global dataflow graph with schedule and binding (unrolled)"', dot)
        self.assertIn("cc 0", dot)
        self.assertIn("reg 1", dot)
        self.assertIn("->", dot)

    def test_dfgsb_dot_separates_unbound_ctrl_ops_in_same_cycle(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design pred
                stage sched
                schedule kind=control_steps

                region proc_pred kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = sel c, a, b : i32 class=CTRL ii=0 delay=0 start=1 end=1
                  node v2 = ret x class=CTRL ii=0 delay=0 start=1 end=1
                  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                }
                """
            )
        )

        dot = bind_dump_to_dot(bind_design, ("dfgsb",))
        self.assertIn('label="v1\\nsel\\nctrl"', dot)
        self.assertIn('label="v2\\nret\\nctrl"', dot)
        self.assertNotIn('label="v1 sel\\nv2 ret"', dot)

    def test_dfgsb_dot_renders_constant_and_input_sources(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design source_nodes
                stage sched
                input  A : memref<i32>
                input  idx : i32
                input  c : i1
                output result : i32
                schedule kind=control_steps

                region proc_source_nodes kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = load A, idx : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = sel c, x, 0:i32 : i32 class=CTRL ii=0 delay=0 start=1 end=1
                  node v3 = ret v2 class=CTRL ii=0 delay=0 start=1 end=1
                  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                  edge data v3 -> v4

                  map v1 <- x
                }
                """
            )
        )

        dot = bind_dump_to_dot(bind_design, ("dfgsb",))
        self.assertIn('label="A"', dot)
        self.assertIn('style="filled,bold"', dot)
        self.assertIn('label="0:i32"', dot)
        self.assertIn('style="filled,dashed"', dot)

    def test_dfgsb_unroll_renders_input_and_constant_sources_per_consumption(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design source_nodes_unroll
                stage sched
                input  A : i32
                input  c : i1
                schedule kind=control_steps

                region proc_source_nodes_unroll kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = sel c, A, 0:i32 : i32 class=CTRL ii=0 delay=0 start=1 end=1
                  node v2 = sel c, A, 0:i32 : i32 class=CTRL ii=0 delay=0 start=2 end=2
                  node v3 = ret v2 class=CTRL ii=0 delay=0 start=3 end=3
                  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                  edge data v3 -> v4
                }
                """
            )
        )

        dot = bind_dump_to_dot(bind_design, ("dfgsb_unroll",))
        self.assertGreaterEqual(dot.count('label="0:i32"'), 2)
        self.assertGreaterEqual(dot.count('label="A"'), 2)

    def test_dfgsb_dot_renders_output_sink_for_return(self) -> None:
        bind_design = lower_sched_to_bind(
            parse_uhir(
                """
                design out_node
                stage sched
                output result : i32
                schedule kind=control_steps

                region proc_out_node kind=procedure {
                  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
                  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
                  node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
                  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

                  edge data v0 -> v1
                  edge data v1 -> v2
                  edge data v2 -> v3
                }
                """
            )
        )

        dot = bind_dump_to_dot(bind_design, ("dfgsb",))
        self.assertIn('label="v1"', dot)
        self.assertIn('style="filled,bold"', dot)
        self.assertIn('-> "dfgsb_proc_out_node_out_0"', dot)

    def test_lower_sched_to_bind_reuses_resources_across_mutually_exclusive_branch_sgus(self) -> None:
        sched_design = parse_uhir(
            """
            design branch_share
            stage sched
            schedule kind=hierarchical

            region proc_branch_share kind=procedure {
              region_ref bb_true
              region_ref bb_false
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = lt a, b : i1 class=EWMS ii=1 delay=1 start=0 end=0
              node v2 = branch v1 true_child=bb_true false_child=bb_false class=CTRL ii=0 delay=1 start=1 end=1
              node v3 = ret x class=CTRL ii=0 delay=0 start=3 end=3
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4

              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
            }

            region bb_true kind=basic parent=proc_branch_share {
              node t0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node t1 = add a, b : i32 class=EWMS ii=1 delay=1 start=2 end=2
              node t2 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

              edge data t0 -> t1
              edge data t1 -> t2
            }

            region bb_false kind=basic parent=proc_branch_share {
              node f0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node f1 = add c, d : i32 class=EWMS ii=1 delay=1 start=2 end=2
              node f2 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

              edge data f0 -> f1
              edge data f1 -> f2
            }
            """
        )

        bind_design = lower_sched_to_bind(sched_design)
        true_region = bind_design.get_region("bb_true")
        false_region = bind_design.get_region("bb_false")
        assert true_region is not None
        assert false_region is not None

        true_add = next(node for node in true_region.nodes if node.id == "t1")
        false_add = next(node for node in false_region.nodes if node.id == "f1")
        self.assertEqual(true_add.attributes["bind"], false_add.attributes["bind"])
        self.assertEqual([resource.id for resource in bind_design.resources if resource.kind == "fu"], ["ewms0"])

    def test_flattened_binding_rejects_non_static_loop_design(self) -> None:
        sched_design = parse_uhir(
            """
            design dynamic_loop
            stage sched
            schedule kind=hierarchical

            region proc_dynamic_loop kind=procedure {
              region_ref loop_header_1
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = loop child=loop_header_1 class=CTRL ii=0 delay=4 start=0 end=4
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=5 end=5

              edge data v0 -> v1
              edge data v1 -> v2
            }

            region loop_header_1 kind=loop parent=proc_dynamic_loop {
              node h0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node h1 = branch c true_child=loop_body_1 false_child=loop_exit_1 class=CTRL ii=0 delay=1 start=1 end=1
              node h2 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

              edge data h0 -> h1
              edge data h1 -> h2
            }

            region loop_body_1 kind=basic parent=loop_header_1 {
              node b0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node b1 = add a, b : i32 class=EWMS ii=1 delay=1 start=2 end=2
              node b2 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

              edge data b0 -> b1
              edge data b1 -> b2
            }

            region loop_exit_1 kind=basic parent=loop_header_1 {
              node e0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node e1 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1

              edge data e0 -> e1
            }
            """
        )

        with self.assertRaisesRegex(ValueError, "fully static design"):
            lower_sched_to_bind(sched_design, binder=LeftEdgeBinder(flatten=True))
