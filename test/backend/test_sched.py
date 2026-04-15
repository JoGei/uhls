from __future__ import annotations

import unittest

from uhls.backend.hls import SGUScheduleResult, lower_alloc_to_sched
from uhls.backend.hls.uhir import (
    ExecutabilityGraph,
    create_builtin_gopt_pass,
    lower_module_to_seq,
    lower_seq_to_alloc,
    parse_uhir,
    run_gopt_passes,
)
from uhls.backend.hls.uhir.timing import parse_timing_expr
from uhls.frontend import lower_source_to_uir
from uhls.middleend.passes.opt import CanonicalizeLoopsPass, ConstPropPass, CopyPropPass, DCEPass, UnrollLoopsPass
from uhls.middleend.passes.util import PassManager
from uhls.middleend.uir import BinaryOp, Block, CallOp, CompareOp, CondBranchOp, Function, Module, Parameter, ReturnOp


def _full_executability_graph() -> ExecutabilityGraph:
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
        functional_units=("FU_GENERIC", "FU_FAST_ADD"),
        operations=operations,
        edges=tuple(("FU_GENERIC", operation, 1, 1) for operation in operations if operation != "add")
        + (("FU_GENERIC", "add", 2, 2), ("FU_FAST_ADD", "add", 1, 1)),
    )


class _FixedScheduler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def schedule_sgu(self, region) -> SGUScheduleResult:
        self.calls.append(region.id)
        starts = {node.id: (0, 0) for node in region.nodes}
        if region.id == "proc_callee":
            for node in region.nodes:
                if node.opcode == "add":
                    starts[node.id] = (1, 1)
                elif node.opcode == "ret":
                    starts[node.id] = (2, 2)
            return SGUScheduleResult(region.id, starts, latency=3)
        if region.id == "proc_caller":
            for node in region.nodes:
                if node.opcode == "call":
                    starts[node.id] = (5, 7)
                elif node.opcode == "ret":
                    starts[node.id] = (9, 9)
            return SGUScheduleResult(region.id, starts, latency=10)
        raise AssertionError(f"unexpected region {region.id}")


class _SymbolicScheduler:
    def schedule_sgu(self, region) -> SGUScheduleResult:
        starts = {node.id: (parse_timing_expr("t0"), parse_timing_expr("t1")) for node in region.nodes}
        return SGUScheduleResult(
            region.id,
            starts,
            latency=parse_timing_expr("L"),
            initiation_interval=parse_timing_expr("II"),
            steps=(parse_timing_expr("t0"), parse_timing_expr("t1")),
        )


class SchedulingLoweringTests(unittest.TestCase):
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

    def test_lower_alloc_to_sched_uses_builtin_asap_for_flat_regions(self) -> None:
        alloc_design = lower_seq_to_alloc(
            lower_module_to_seq(
                Module(
                    name="demo",
                    functions=[
                        Function(
                            name="add1",
                            params=[Parameter("x", "i32")],
                            return_type="i32",
                            blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
                        )
                    ],
                )
            ),
            executability_graph=_full_executability_graph(),
        )

        sched_design = lower_alloc_to_sched(alloc_design)
        proc = sched_design.get_region("proc_add1")
        self.assertIsNotNone(proc)
        assert proc is not None
        add_node = next(node for node in proc.nodes if node.opcode == "add")
        ret_node = next(node for node in proc.nodes if node.opcode == "ret")

        self.assertEqual(sched_design.stage, "sched")
        self.assertEqual(sched_design.schedule.kind, "hierarchical")
        self.assertEqual(add_node.attributes["start"], 0)
        self.assertEqual(add_node.attributes["end"], 0)
        self.assertEqual(ret_node.attributes["start"], 1)
        self.assertEqual(ret_node.attributes["end"], 1)
        self.assertEqual(proc.latency, 2)

    def test_lower_alloc_to_sched_uses_injected_flat_scheduler(self) -> None:
        alloc_design = lower_seq_to_alloc(
            lower_module_to_seq(
                Module(
                    name="demo",
                    functions=[
                        Function(
                            name="callee",
                            params=[Parameter("x", "i32")],
                            return_type="i32",
                            blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
                        ),
                        Function(
                            name="caller",
                            params=[Parameter("x", "i32")],
                            return_type="i32",
                            blocks=[Block("entry", instructions=[CallOp("callee", ["x"], dest="z", type="i32")], terminator=ReturnOp("z"))],
                        ),
                    ],
                ),
                top="caller",
            ),
            executability_graph=_full_executability_graph(),
        )

        scheduler = _FixedScheduler()
        sched_design = lower_alloc_to_sched(alloc_design, scheduler=scheduler)

        caller = sched_design.get_region("proc_caller")
        callee = sched_design.get_region("proc_callee")
        self.assertEqual(scheduler.calls, ["proc_callee", "proc_caller"])
        self.assertIsNotNone(caller)
        self.assertIsNotNone(callee)
        assert caller is not None
        assert callee is not None

        call_node = next(node for node in caller.nodes if node.opcode == "call")
        ret_node = next(node for node in caller.nodes if node.opcode == "ret")
        callee_add = next(node for node in callee.nodes if node.opcode == "add")

        self.assertEqual(call_node.attributes["delay"], 3)
        self.assertEqual(call_node.attributes["ii"], 3)
        self.assertEqual(call_node.attributes["start"], 5)
        self.assertEqual(call_node.attributes["end"], 7)
        self.assertEqual(ret_node.attributes["start"], 9)
        self.assertEqual(callee_add.attributes["start"], 6)
        self.assertEqual(callee_add.attributes["end"], 6)
        self.assertEqual(callee.latency, 3)
        self.assertEqual(caller.latency, 10)

    def test_lower_alloc_to_sched_accepts_symbolic_schedule_result_for_flat_region(self) -> None:
        alloc_design = lower_seq_to_alloc(
            lower_module_to_seq(
                Module(
                    name="demo",
                    functions=[
                        Function(
                            name="add1",
                            params=[Parameter("x", "i32")],
                            return_type="i32",
                            blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
                        )
                    ],
                )
            ),
            executability_graph=_full_executability_graph(),
        )

        sched_design = lower_alloc_to_sched(alloc_design, scheduler=_SymbolicScheduler())
        proc = sched_design.get_region("proc_add1")
        self.assertIsNotNone(proc)
        assert proc is not None
        add_node = next(node for node in proc.nodes if node.opcode == "add")

        self.assertEqual(str(add_node.attributes["start"]), "t0")
        self.assertEqual(str(add_node.attributes["end"]), "t1")
        self.assertEqual(tuple(str(value) for value in proc.steps), ("t0", "t1"))
        self.assertEqual(str(proc.latency), "L")
        self.assertEqual(str(proc.initiation_interval), "II")

    def test_lower_alloc_to_sched_accepts_statically_bounded_loops(self) -> None:
        alloc_design = lower_seq_to_alloc(
            self._infer_static(
                lower_module_to_seq(
                lower_source_to_uir(
                    """
                    int32_t dot4(int32_t A[4], int32_t B[4]) {
                        int32_t i;
                        int32_t sum = 0;
                        for (i = 0; i < 4; i = i + 1) {
                            sum = sum + A[i] * B[i];
                        }
                        return sum;
                    }
                    """
                ),
                top="dot4",
            ),
            ),
            executability_graph=_full_executability_graph(),
        )

        sched_design = lower_alloc_to_sched(alloc_design)
        proc = sched_design.get_region("proc_dot4")
        loop_region = next((region for region in sched_design.regions if region.kind == "loop"), None)
        body_region = next((region for region in sched_design.regions if region.kind == "body"), None)
        self.assertIsNotNone(proc)
        self.assertIsNotNone(loop_region)
        self.assertIsNotNone(body_region)
        assert proc is not None
        assert loop_region is not None
        assert body_region is not None

        loop_node = next(node for node in proc.nodes if node.opcode == "loop")
        loop_branch = next(node for node in loop_region.nodes if node.opcode == "branch")
        loop_compare = next(node for node in loop_region.nodes if node.opcode == "lt")
        body_mul = next(node for node in body_region.nodes if node.opcode == "mul")

        self.assertEqual(loop_node.attributes.get("static_trip_count"), 4)
        self.assertNotIn("static_trip_count", loop_branch.attributes)
        self.assertIsNotNone(loop_node.attributes.get("start"))
        self.assertIsNotNone(loop_node.attributes.get("end"))
        self.assertIsNotNone(loop_region.latency)
        self.assertIsNotNone(body_region.latency)
        self.assertEqual(loop_branch.attributes["delay"], max(body_region.latency or 0, 0))
        self.assertEqual(loop_compare.attributes["start"], 0)
        self.assertEqual(loop_compare.attributes["end"], 0)
        self.assertEqual(loop_branch.attributes["start"], 1)
        self.assertEqual(loop_branch.attributes["end"], loop_region.latency - 1)
        self.assertEqual(body_mul.attributes["start"], 2)
        self.assertEqual(body_mul.attributes["end"], 2)
        iter_latency = loop_node.attributes["iter_latency"]
        iter_initiation_interval = loop_node.attributes["iter_initiation_interval"]
        iter_ramp_down = loop_node.attributes["iter_ramp_down"]
        self.assertIsInstance(iter_latency, int)
        self.assertIsInstance(iter_initiation_interval, int)
        self.assertIsInstance(iter_ramp_down, int)
        self.assertEqual(iter_latency, iter_initiation_interval)
        self.assertEqual(iter_ramp_down, 0)
        self.assertEqual(iter_initiation_interval, loop_region.latency)
        self.assertGreaterEqual(iter_initiation_interval, body_region.latency)
        self.assertEqual(loop_node.attributes["delay"], 4 * iter_initiation_interval + iter_ramp_down)
        self.assertEqual(loop_node.attributes["ii"], iter_initiation_interval)
        self.assertEqual(loop_node.attributes["timing"], "static")
        self.assertNotIn("completion", loop_node.attributes)
        self.assertNotIn("ready", loop_node.attributes)
        self.assertNotIn("handshake", loop_node.attributes)
        self.assertNotIn("continue_condition", loop_node.attributes)
        self.assertNotIn("iterate_when", loop_node.attributes)
        self.assertNotIn("exit_when", loop_node.attributes)

    def test_lower_alloc_to_sched_overlaps_duplicated_body_ops_for_unrolled_do_all_loop(self) -> None:
        source = """
        int32_t add8(int32_t A[8], int32_t B[8], int32_t C[8]) {
            int32_t i;
            for (i = 0; i < 8; i = i + 1) {
                C[i] = A[i] + B[i];
            }
            return 0;
        }
        """
        optimized_module = PassManager(
            [
                UnrollLoopsPass("1", 2),
                CanonicalizeLoopsPass(),
                ConstPropPass(),
                CopyPropPass(),
                DCEPass(),
            ]
        ).run(lower_source_to_uir(source))
        seq_design = run_gopt_passes(
            lower_module_to_seq(optimized_module, top="add8"),
            [
                create_builtin_gopt_pass("infer_loops"),
                create_builtin_gopt_pass("translate_loop_dialect"),
            ],
        )
        sched_design = lower_alloc_to_sched(
            lower_seq_to_alloc(seq_design, executability_graph=_full_executability_graph())
        )

        body_region = sched_design.get_region("loop_body_add8_for_header_1")
        self.assertIsNotNone(body_region)
        assert body_region is not None

        producer_by_source = {}
        nodes_by_id = {node.id: node for node in body_region.nodes}
        for mapping in body_region.mappings:
            node = nodes_by_id.get(mapping.node_id)
            if node is not None:
                producer_by_source[mapping.source_id] = node

        first_load_a = producer_by_source["t1_0"]
        first_load_b = producer_by_source["t2_0"]
        first_add = producer_by_source["t3_0"]
        first_next_i = producer_by_source["t4_0"]
        second_load_a = producer_by_source["t1_0__u1"]
        second_load_b = producer_by_source["t2_0__u1"]
        second_add = producer_by_source["t3_0__u1"]

        self.assertEqual(first_load_a.opcode, "load")
        self.assertEqual(first_load_b.opcode, "load")
        self.assertEqual(first_add.opcode, "add")
        self.assertEqual(second_load_a.opcode, "load")
        self.assertEqual(second_load_b.opcode, "load")
        self.assertEqual(second_add.opcode, "add")
        self.assertEqual(first_load_a.attributes["start"], 1)
        self.assertEqual(first_load_b.attributes["start"], 1)
        self.assertEqual(first_next_i.attributes["start"], 1)
        self.assertEqual(first_add.attributes["start"], 2)
        self.assertEqual(second_load_a.attributes["start"], 2)
        self.assertEqual(second_load_b.attributes["start"], 2)
        self.assertEqual(second_add.attributes["start"], 3)
        self.assertEqual(second_load_a.attributes["start"], first_add.attributes["start"])
        self.assertEqual(second_load_b.attributes["start"], first_add.attributes["start"])
        self.assertGreater(second_add.attributes["start"], second_load_a.attributes["start"])

    def test_lower_alloc_to_sched_uses_max_child_latency_for_branches(self) -> None:
        alloc_design = lower_seq_to_alloc(
            lower_module_to_seq(
                Module(
                    name="demo",
                    functions=[
                        Function(
                            name="choose",
                            params=[Parameter("x", "i32"), Parameter("y", "i32")],
                            return_type="i32",
                            blocks=[
                                Block(
                                    "entry",
                                    instructions=[CompareOp("lt", "cond", "x", "y", "i1")],
                                    terminator=CondBranchOp("cond", "then_blk", "else_blk"),
                                ),
                                Block(
                                    "then_blk",
                                    instructions=[BinaryOp("add", "then_val", "i32", "x", 1)],
                                    terminator=ReturnOp("then_val"),
                                ),
                                Block(
                                    "else_blk",
                                    instructions=[
                                        BinaryOp("add", "tmp", "i32", "x", "y"),
                                        BinaryOp("add", "else_val", "i32", "tmp", 1),
                                    ],
                                    terminator=ReturnOp("else_val"),
                                ),
                            ],
                        )
                    ],
                ),
                top="choose",
            ),
            executability_graph=_full_executability_graph(),
        )

        sched_design = lower_alloc_to_sched(alloc_design)
        proc = sched_design.get_region("proc_choose")
        self.assertIsNotNone(proc)
        assert proc is not None

        branch_node = next(node for node in proc.nodes if node.opcode == "branch")
        true_region = sched_design.get_region(branch_node.attributes["true_child"])
        false_region = sched_design.get_region(branch_node.attributes["false_child"])
        self.assertIsNotNone(true_region)
        self.assertIsNotNone(false_region)
        assert true_region is not None
        assert false_region is not None

        self.assertEqual(branch_node.attributes["delay"], max(true_region.latency or 0, false_region.latency or 0))
        self.assertEqual(branch_node.attributes["timing"], "static")
        self.assertNotIn("completion", branch_node.attributes)
        self.assertNotIn("branch_condition", branch_node.attributes)

    def test_lower_alloc_to_sched_uses_symbolic_max_for_dynamic_branch_latency(self) -> None:
        alloc_design = parse_uhir(
            """
            design dyn_branch
            stage alloc

            region proc_dyn_branch kind=procedure {
              region_ref bb_true
              region_ref bb_false
              node v0 = nop role=source class=CTRL ii=0 delay=0
              node v1 = branch c true_child=bb_true false_child=bb_false class=CTRL ii=0 delay=0
              node v2 = nop role=sink class=CTRL ii=0 delay=0
              edge data v0 -> v1
              edge data v1 -> v2
            }

            region bb_true kind=basic parent=proc_dyn_branch {
              node t0 = nop role=source class=CTRL ii=0 delay=0
              node t1 = add a, b : i32 class=FU_FAST_ADD ii=1 delay=1
              node t2 = nop role=sink class=CTRL ii=0 delay=0
              edge data t0 -> t1
              edge data t1 -> t2
            }

            region bb_false kind=basic parent=proc_dyn_branch {
              node f0 = nop role=source class=CTRL ii=0 delay=0
              node f1 = add a, b : i32 class=FU_FAST_ADD ii=1 delay=1
              node f2 = add f1, c : i32 class=FU_FAST_ADD ii=1 delay=1
              node f3 = nop role=sink class=CTRL ii=0 delay=0
              edge data f0 -> f1
              edge data f1 -> f2
              edge data f2 -> f3
            }
            """
        )

        sched_design = lower_alloc_to_sched(alloc_design, scheduler=_SymbolicScheduler())
        proc = sched_design.get_region("proc_dyn_branch")
        assert proc is not None
        branch_node = next(node for node in proc.nodes if node.opcode == "branch")
        self.assertEqual(str(branch_node.attributes["delay"]), "L")
        self.assertEqual(branch_node.attributes["timing"], "symbolic")
        self.assertEqual(branch_node.attributes["completion"], "symb_done_v1")
        self.assertEqual(branch_node.attributes["branch_condition"], "c")
        self.assertNotIn("ready", branch_node.attributes)
        self.assertNotIn("handshake", branch_node.attributes)
        self.assertNotIn("continue_condition", branch_node.attributes)

    def test_lower_alloc_to_sched_uses_symbolic_loop_summary_when_not_static(self) -> None:
        alloc_design = parse_uhir(
            """
            design dyn_loop
            stage alloc

            region proc_dyn_loop kind=procedure {
              region_ref loop_header_1
              node v0 = nop role=source class=CTRL ii=0 delay=0
              node v1 = loop child=loop_header_1 class=CTRL ii=0 delay=0
              node v2 = nop role=sink class=CTRL ii=0 delay=0
              edge data v0 -> v1
              edge data v1 -> v2
            }

            region loop_header_1 kind=loop parent=proc_dyn_loop {
              region_ref loop_body_1
              region_ref loop_exit_1
              node h0 = nop role=source class=CTRL ii=0 delay=0
              node h1 = branch c true_child=loop_body_1 false_child=loop_exit_1 class=CTRL ii=0 delay=0
              node h2 = nop role=sink class=CTRL ii=0 delay=0
              edge data h0 -> h1
              edge data h1 -> h2
            }

            region loop_body_1 kind=body parent=loop_header_1 {
              node b0 = nop role=source class=CTRL ii=0 delay=0
              node b1 = add x, y : i32 class=FU_FAST_ADD ii=1 delay=1
              node b2 = nop role=sink class=CTRL ii=0 delay=0
              edge data b0 -> b1
              edge data b1 -> b2
            }

            region loop_exit_1 kind=empty parent=loop_header_1 {
              node e0 = nop role=source class=CTRL ii=0 delay=0
              node e1 = nop role=sink class=CTRL ii=0 delay=0
              edge data e0 -> e1
            }
            """
        )

        sched_design = lower_alloc_to_sched(alloc_design, scheduler=_SymbolicScheduler())
        proc = sched_design.get_region("proc_dyn_loop")
        assert proc is not None
        loop_node = next(node for node in proc.nodes if node.opcode == "loop")
        self.assertNotIn("iter_latency", loop_node.attributes)
        self.assertNotIn("iter_initiation_interval", loop_node.attributes)
        self.assertNotIn("iter_ramp_down", loop_node.attributes)
        self.assertEqual(str(loop_node.attributes["delay"]), "symb_delay_v1")
        self.assertEqual(str(loop_node.attributes["ii"]), "II")
        self.assertEqual(loop_node.attributes["timing"], "symbolic")
        self.assertEqual(loop_node.attributes["completion"], "symb_done_v1")
        self.assertEqual(loop_node.attributes["ready"], "symb_ready_v1")
        self.assertEqual(loop_node.attributes["handshake"], "ready_done")
        self.assertEqual(loop_node.attributes["continue_condition"], "c")
        self.assertEqual(loop_node.attributes["iterate_when"], "c")
        self.assertEqual(loop_node.attributes["exit_when"], "!c")
        self.assertNotIn("branch_condition", loop_node.attributes)

    def test_lower_alloc_to_sched_uses_symbolic_call_summary_when_callee_is_dynamic(self) -> None:
        alloc_design = parse_uhir(
            """
            design dyn_call
            stage alloc

            region proc_caller kind=procedure {
              region_ref proc_callee
              node v0 = nop role=source class=CTRL ii=0 delay=0
              node v1 = call x child=proc_callee class=CTRL ii=0 delay=0
              node v2 = ret v1 class=CTRL ii=0 delay=0
              node v3 = nop role=sink class=CTRL ii=0 delay=0
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
            }

            region proc_callee kind=procedure {
              node c0 = nop role=source class=CTRL ii=0 delay=0
              node c1 = add x, y : i32 class=FU_FAST_ADD ii=1 delay=1
              node c2 = ret c1 class=CTRL ii=0 delay=0
              node c3 = nop role=sink class=CTRL ii=0 delay=0
              edge data c0 -> c1
              edge data c1 -> c2
              edge data c2 -> c3
            }
            """
        )

        sched_design = lower_alloc_to_sched(alloc_design, scheduler=_SymbolicScheduler())
        caller = sched_design.get_region("proc_caller")
        assert caller is not None
        call_node = next(node for node in caller.nodes if node.opcode == "call")
        self.assertEqual(str(call_node.attributes["delay"]), "symb_delay_v1")
        self.assertEqual(str(call_node.attributes["ii"]), "II")
        self.assertEqual(call_node.attributes["timing"], "symbolic")
        self.assertEqual(call_node.attributes["completion"], "symb_done_v1")
        self.assertEqual(call_node.attributes["ready"], "symb_ready_v1")
        self.assertEqual(call_node.attributes["handshake"], "ready_done")
        self.assertNotIn("branch_condition", call_node.attributes)
        self.assertNotIn("continue_condition", call_node.attributes)

    def test_lower_alloc_to_sched_builtin_asap_accepts_symbolic_hierarchy_delay(self) -> None:
        alloc_design = parse_uhir(
            """
            design dyn_parent
            stage alloc

            region proc_dyn_parent kind=procedure {
              region_ref loop_header_1
              node v0 = nop role=source class=CTRL ii=0 delay=0
              node v1 = loop child=loop_header_1 class=CTRL ii=0 delay=0
              node v2 = add x, y : i32 class=FU_FAST_ADD ii=1 delay=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
            }

            region loop_header_1 kind=loop parent=proc_dyn_parent {
              region_ref loop_body_1
              region_ref loop_exit_1
              node h0 = nop role=source class=CTRL ii=0 delay=0
              node h1 = branch c true_child=loop_body_1 false_child=loop_exit_1 class=CTRL ii=0 delay=0
              node h2 = nop role=sink class=CTRL ii=0 delay=0
              edge data h0 -> h1
              edge data h1 -> h2
            }

            region loop_body_1 kind=body parent=loop_header_1 {
              node b0 = nop role=source class=CTRL ii=0 delay=0
              node b1 = add x, y : i32 class=FU_FAST_ADD ii=1 delay=1
              node b2 = nop role=sink class=CTRL ii=0 delay=0
              edge data b0 -> b1
              edge data b1 -> b2
            }

            region loop_exit_1 kind=empty parent=loop_header_1 {
              node e0 = nop role=source class=CTRL ii=0 delay=0
              node e1 = nop role=sink class=CTRL ii=0 delay=0
              edge data e0 -> e1
            }
            """
        )

        sched_design = lower_alloc_to_sched(alloc_design)
        proc = sched_design.get_region("proc_dyn_parent")
        assert proc is not None
        loop_node = next(node for node in proc.nodes if node.opcode == "loop")
        add_node = next(node for node in proc.nodes if node.opcode == "add")
        self.assertEqual(str(loop_node.attributes["delay"]), "symb_delay_v1")
        self.assertEqual(str(loop_node.attributes["end"]), "symb_delay_v1 - 1")
        self.assertEqual(str(add_node.attributes["start"]), "symb_delay_v1")

    def test_lower_alloc_to_sched_rejects_non_sink_leaf_nodes(self) -> None:
        alloc_design = parse_uhir(
            """design bad
stage alloc

region proc_bad kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0
  node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1
  node v2 = nop role=sink class=CTRL ii=0 delay=0
}
"""
        )

        with self.assertRaisesRegex(ValueError, "must contain exactly one leaf node|must be the sink nop"):
            lower_alloc_to_sched(alloc_design)
