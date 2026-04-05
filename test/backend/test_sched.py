from __future__ import annotations

import unittest

from uhls.backend.hls import SGUScheduleResult, lower_alloc_to_sched
from uhls.backend.uhir import (
    ExecutabilityGraph,
    create_builtin_gopt_pass,
    lower_module_to_seq,
    lower_seq_to_alloc,
    parse_uhir,
    run_gopt_passes,
)
from uhls.frontend import lower_source_to_uir
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


class SchedulingLoweringTests(unittest.TestCase):
    @staticmethod
    def _infer_static(design):
        return run_gopt_passes(design, [create_builtin_gopt_pass("infer_static")])

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
        self.assertEqual(call_node.attributes["start"], 5)
        self.assertEqual(call_node.attributes["end"], 7)
        self.assertEqual(ret_node.attributes["start"], 9)
        self.assertEqual(callee_add.attributes["start"], 6)
        self.assertEqual(callee_add.attributes["end"], 6)
        self.assertEqual(callee.latency, 3)
        self.assertEqual(caller.latency, 10)

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
        self.assertEqual(loop_compare.attributes["start"], 1)
        self.assertEqual(loop_branch.attributes["start"], 2)
        self.assertEqual(body_mul.attributes["start"], 3)
        self.assertEqual(body_mul.attributes["end"], 3)
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
