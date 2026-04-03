from __future__ import annotations

import unittest

import uhls.middleend.passes
from uhls.frontend import lower_source_to_uir
from uhls.interpreter import run_uir
from uhls.middleend.passes.analyze import LivenessInfo, build_dfg, compute_dominators, detect_loops, dfg_pass, liveliness, liveliness_pass
from uhls.middleend.passes.opt import SimplifyCFGPass, inline_calls, simplify_cfg_function
from uhls.middleend.passes.opt.const_prop import const_prop_function
from uhls.middleend.passes.opt.copy_prop import copy_prop_function
from uhls.middleend.passes.opt.cse import cse_function
from uhls.middleend.passes.opt.dce import dce_function
from uhls.middleend.passes.opt.prune_functions import prune_functions_module
from uhls.middleend.passes.util import PassContext, PassManager
from uhls.middleend.passes.util.dot import to_basic_block_dfg_dot, to_cdfg_dot, to_dot
from uhls.middleend.uir import (
    BinaryOp,
    Block,
    BranchOp,
    CallOp,
    CompareOp,
    CondBranchOp,
    ConstOp,
    Function,
    IncomingValue,
    Literal,
    LoadOp,
    Module,
    Parameter,
    ParamOp,
    PhiOp,
    PrintOp,
    ReturnOp,
    ScalarType,
    StoreOp,
    UnaryOp,
    Variable,
    parse_module,
    pretty,
    verify_function,
    verify_module,
)


class _RecordPass:
    name = "record"

    def run(self, ir: Function, context: PassContext) -> Function:
        context.data["seen"] = ir.name
        return ir


class MiddleendPackageTests(unittest.TestCase):
    """Coverage for the middle-end package split."""

    def test_middleend_passes_package_exports_split_subpackages(self) -> None:
        self.assertTrue(hasattr(uhls.middleend.passes, "analyze"))
        self.assertTrue(hasattr(uhls.middleend.passes, "opt"))
        self.assertTrue(hasattr(uhls.middleend.passes, "util"))

    def test_middleend_opt_package_exports_transform_and_optimization_entry_points(self) -> None:
        self.assertTrue(hasattr(uhls.middleend.passes.opt, "DCEPass"))
        self.assertTrue(hasattr(uhls.middleend.passes.opt, "ConstPropPass"))
        self.assertTrue(hasattr(uhls.middleend.passes.opt, "CopyPropPass"))
        self.assertTrue(hasattr(uhls.middleend.passes.opt, "CSEPass"))
        self.assertTrue(hasattr(uhls.middleend.passes.opt, "InlineCallsPass"))
        self.assertTrue(hasattr(uhls.middleend.passes.opt, "PruneFunctionsPass"))
        self.assertTrue(hasattr(uhls.middleend.passes.opt, "SimplifyCFGPass"))


class UIRPackageTests(unittest.TestCase):
    """Coverage for the canonical UIR package."""

    def test_uir_function_pretty_prints_and_verifies(self) -> None:
        i1 = ScalarType("i1")
        i32 = ScalarType("i32")
        function = Function(
            name="uir_select",
            params=[Parameter("sel", i1), Parameter("idx", "u8")],
            return_type=i32,
            blocks=[
                Block("entry", terminator=CondBranchOp("sel", "then_blk", "else_blk")),
                Block(
                    "then_blk",
                    instructions=[
                        LoadOp("tmp", i32, "A", "idx"),
                        BinaryOp("add", "x", i32, "tmp", 1),
                    ],
                    terminator=BranchOp("merge_blk"),
                ),
                Block(
                    "else_blk",
                    instructions=[
                        LoadOp("tmp", i32, "B", "idx"),
                        BinaryOp("sub", "x", i32, "tmp", 1),
                    ],
                    terminator=BranchOp("merge_blk"),
                ),
                Block(
                    "merge_blk",
                    instructions=[StoreOp("OUT", 0, "x")],
                    terminator=ReturnOp("x"),
                ),
            ],
        )

        verify_function(function)
        rendered = pretty(function)

        self.assertIn("func uir_select(sel:i1, idx:u8) -> i32", rendered)
        self.assertIn("cbr sel, then_blk, else_blk", rendered)
        self.assertIn("store OUT[0], x", rendered)

    def test_ir_function_runs_in_uir_interpreter(self) -> None:
        function = Function(
            name="uir_select",
            params=[Parameter("sel", "i1"), Parameter("idx", "u8")],
            return_type="i32",
            blocks=[
                Block("entry", terminator=CondBranchOp("sel", "then_blk", "else_blk")),
                Block(
                    "then_blk",
                    instructions=[
                        LoadOp("tmp", "i32", "A", "idx"),
                        BinaryOp("add", "x", "i32", "tmp", 1),
                    ],
                    terminator=BranchOp("merge_blk"),
                ),
                Block(
                    "else_blk",
                    instructions=[
                        LoadOp("tmp", "i32", "B", "idx"),
                        BinaryOp("sub", "x", "i32", "tmp", 1),
                    ],
                    terminator=BranchOp("merge_blk"),
                ),
                Block(
                    "merge_blk",
                    instructions=[StoreOp("OUT", 0, "x")],
                    terminator=ReturnOp("x"),
                ),
            ],
        )

        result = run_uir(
            function,
            {"sel": 1, "idx": 1},
            arrays={
                "A": {"data": [10, 20], "element_type": "i32"},
                "B": {"data": [30, 40], "element_type": "i32"},
                "OUT": {"size": 1, "element_type": "i32"},
            },
        )

        self.assertEqual(result.return_value, 21)
        self.assertEqual(result.state.memory.snapshot()["OUT"], [21])

    def test_interpreter_resolves_ir_value_operands(self) -> None:
        function = Function(
            name="value_operands",
            params=[Parameter("idx", "u8")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        LoadOp("tmp", "i32", "A", Variable("idx", "u8")),
                        BinaryOp("add", "x", "i32", Variable("tmp", "i32"), Literal(1, "i32")),
                        StoreOp("OUT", Literal(0, "u8"), Variable("x", "i32")),
                    ],
                    terminator=ReturnOp(Variable("x", "i32")),
                ),
            ],
        )

        result = run_uir(
            function,
            {"idx": 1},
            arrays={
                "A": {"data": [10, 20], "element_type": ScalarType("i32")},
                "OUT": {"size": 1, "element_type": ScalarType("i32")},
            },
        )

        self.assertEqual(result.return_value, 21)
        self.assertEqual(result.state.memory.snapshot()["OUT"], [21])

    def test_uir_module_verifies_and_runs(self) -> None:
        function = Function(
            name="uir_phi_select",
            params=[Parameter("sel_0", "i1"), Parameter("a_0", "i32"), Parameter("b_0", "i32")],
            return_type="i32",
            blocks=[
                Block("entry", terminator=CondBranchOp("sel_0", "then_blk", "else_blk")),
                Block(
                    "then_blk",
                    instructions=[BinaryOp("add", "x_1", "i32", "a_0", 3)],
                    terminator=BranchOp("merge_blk"),
                ),
                Block(
                    "else_blk",
                    instructions=[BinaryOp("sub", "x_2", "i32", "b_0", 5)],
                    terminator=BranchOp("merge_blk"),
                ),
                Block(
                    "merge_blk",
                    instructions=[
                        PhiOp(
                            "x_3",
                            "i32",
                            [IncomingValue("then_blk", "x_1"), IncomingValue("else_blk", "x_2")],
                        ),
                        BinaryOp("mul", "y_0", "i32", "x_3", 2),
                    ],
                    terminator=ReturnOp("y_0"),
                ),
            ],
        )
        module = Module(name="demo", functions=[function])

        verify_module(module, require_ssa=True)
        rendered = pretty(module)
        result = run_uir(function, {"sel_0": 0, "a_0": 7, "b_0": 100})

        self.assertIn("module demo", rendered)
        self.assertIn("x_3:i32 = phi(then_blk: x_1, else_blk: x_2)", rendered)
        self.assertEqual(result.return_value, 190)

    def test_lowered_call_syntax_parses_and_pretty_prints(self) -> None:
        module = parse_module(
            """func foo(a:i32, b:i32, c:i32) -> i32

block entry:
    t0:i32 = add a, b
    t1:i32 = add t0, c
    ret t1

func caller(x:i32, y:i32, z:i32) -> i32

block entry:
    param 0, x
    param 1, y
    param 2, z
    out:i32 = call foo, 3
    ret out
"""
        )

        verify_module(module, allow_param=True)
        rendered = pretty(module)
        caller = module.get_function("caller")
        self.assertIsNotNone(caller)
        assert caller is not None
        self.assertIsInstance(caller.blocks[0].instructions[0], ParamOp)
        self.assertIsInstance(caller.blocks[0].instructions[3], CallOp)
        self.assertEqual(caller.blocks[0].instructions[3].arg_count, 3)
        self.assertIn("out:i32 = call foo, 3", rendered)


class PassManagerTests(unittest.TestCase):
    """Coverage for the shared pass-manager framework."""

    def test_pass_manager_runs_pipeline_and_collects_analyses(self) -> None:
        function = Function(
            name="cleanup",
            params=[],
            return_type="i32",
            blocks=[
                Block("entry", terminator=BranchOp("jump")),
                Block("jump", terminator=BranchOp("exit")),
                Block("exit", instructions=[BinaryOp("add", "x", "i32", 2, 3)], terminator=ReturnOp("x")),
            ],
        )

        manager = PassManager([_RecordPass(), simplify_cfg_function, liveliness_pass()])
        result = manager.run_with_context(function)

        self.assertEqual(result.output.blocks[0].label, "entry")
        self.assertEqual(result.context.data["seen"], "cleanup")
        self.assertEqual(result.context.history, ["record", "simplify_cfg_function", "liveliness"])
        self.assertIsInstance(result.context.analyses["liveliness"], LivenessInfo)

    def test_liveliness_alias_reports_branch_live_out_values(self) -> None:
        function = Function(
            name="branch_live",
            params=[Parameter("sel", "i1"), Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block("entry", terminator=CondBranchOp("sel", "then_blk", "else_blk")),
                Block(
                    "then_blk",
                    instructions=[BinaryOp("add", "x", "i32", "a", 1)],
                    terminator=BranchOp("merge_blk"),
                ),
                Block(
                    "else_blk",
                    instructions=[BinaryOp("sub", "x", "i32", "b", 1)],
                    terminator=BranchOp("merge_blk"),
                ),
                Block("merge_blk", terminator=ReturnOp("x")),
            ],
        )

        info = liveliness(function)

        self.assertIn("x", info.live_out["then_blk"])
        self.assertIn("x", info.live_out["else_blk"])
        self.assertIn("a", info.live_in["then_blk"])
        self.assertIn("b", info.live_in["else_blk"])

    def test_existing_pass_objects_work_in_pass_manager_pipeline(self) -> None:
        function = Function(
            name="select",
            params=[Parameter("sel", "i1"), Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block("entry", terminator=CondBranchOp("sel", "then_blk", "else_blk")),
                Block(
                    "then_blk",
                    instructions=[BinaryOp("add", "x", "i32", "a", 1)],
                    terminator=BranchOp("merge_blk"),
                ),
                Block(
                    "else_blk",
                    instructions=[BinaryOp("sub", "x", "i32", "b", 1)],
                    terminator=BranchOp("merge_blk"),
                ),
                Block("merge_blk", terminator=ReturnOp("x")),
            ],
        )

        result = PassManager([SimplifyCFGPass(), liveliness_pass()]).run_with_context(function)

        self.assertEqual(result.context.history, ["simplify_cfg", "liveliness"])
        self.assertIsInstance(result.context.analyses["liveliness"], LivenessInfo)


class CFGPassTests(unittest.TestCase):
    """Coverage for CFG graph utilities and passes."""

    def test_cfg_graph_and_dominators_capture_branch_merge(self) -> None:
        function = Function(
            name="select",
            params=[Parameter("sel", "i1"), Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block("entry", terminator=CondBranchOp("sel", "then_blk", "else_blk")),
                Block(
                    "then_blk",
                    instructions=[BinaryOp("add", "x", "i32", "a", 1)],
                    terminator=BranchOp("merge_blk"),
                ),
                Block(
                    "else_blk",
                    instructions=[BinaryOp("sub", "x", "i32", "b", 1)],
                    terminator=BranchOp("merge_blk"),
                ),
                Block("merge_blk", terminator=ReturnOp("x")),
            ],
        )

        cfg = uhls.middleend.passes.analyze.build_cfg(function)
        dom = compute_dominators(function, cfg)

        self.assertEqual(cfg.successors["entry"], {"then_blk", "else_blk"})
        self.assertEqual(cfg.predecessors["merge_blk"], {"then_blk", "else_blk"})
        self.assertEqual(dom.immediate_dominators["merge_blk"], "entry")
        self.assertEqual(dom.frontiers["then_blk"], {"merge_blk"})
        self.assertEqual(dom.frontiers["else_blk"], {"merge_blk"})

    def test_loop_detection_and_dot_rendering_cover_lowered_frontend_cfg(self) -> None:
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
        function = lower_source_to_uir(source).functions[0]
        loops = detect_loops(function)
        dot = to_dot(function)

        self.assertEqual(len(loops), 1)
        self.assertIn("for_header_1", loops[0].body)
        self.assertIn("for_body_2", loops[0].body)
        self.assertIn('digraph "dot4"', dot)
        self.assertIn('"for_body_2":s -> "for_header_1":n;', dot)

    def test_simplify_cfg_pass_works_in_pass_manager(self) -> None:
        function = Function(
            name="cleanup",
            params=[],
            return_type="i32",
            blocks=[
                Block("entry", terminator=BranchOp("jump")),
                Block("jump", terminator=BranchOp("work")),
                Block("work", instructions=[BinaryOp("add", "x", "i32", 2, 3)], terminator=ReturnOp("x")),
            ],
        )

        result = PassManager([SimplifyCFGPass()]).run_with_context(function)

        self.assertEqual(result.context.history, ["simplify_cfg"])
        self.assertEqual([block.label for block in result.output.blocks], ["entry"])


class DFGPassTests(unittest.TestCase):
    """Coverage for block-local DFG analysis."""

    def test_build_dfg_tracks_scalar_dependencies_within_one_block(self) -> None:
        function = Function(
            name="chain",
            params=[Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        BinaryOp("add", "t0", "i32", "a", "b"),
                        BinaryOp("mul", "t1", "i32", "t0", 2),
                        BinaryOp("sub", "t2", "i32", "t1", "t0"),
                    ],
                    terminator=ReturnOp("t2"),
                )
            ],
        )

        info = build_dfg(function)
        graph = info.blocks["entry"]
        edge_set = {(edge.source, edge.target, edge.kind, edge.label) for edge in graph.edges}

        self.assertEqual([node.id for node in graph.nodes], ["entry:0", "entry:1", "entry:2", "entry:term"])
        self.assertIn(("entry:0", "entry:1", "value", "t0"), edge_set)
        self.assertIn(("entry:0", "entry:2", "value", "t0"), edge_set)
        self.assertIn(("entry:1", "entry:2", "value", "t1"), edge_set)
        self.assertIn(("entry:2", "entry:term", "value", "t2"), edge_set)
        self.assertNotIn(("entry:2", "entry:term", "sink", ""), edge_set)

    def test_build_dfg_adds_sink_edges_from_leaf_nodes_to_unconditional_terminator(self) -> None:
        function = Function(
            name="chain",
            params=[Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        BinaryOp("add", "t0", "i32", "a", "b"),
                        BinaryOp("mul", "t1", "i32", "t0", 2),
                    ],
                    terminator=ReturnOp(),
                )
            ],
        )

        graph = build_dfg(function).blocks["entry"]
        edge_set = {(edge.source, edge.target, edge.kind, edge.label) for edge in graph.edges}

        self.assertIn(("entry:1", "entry:term", "sink", ""), edge_set)

    def test_build_dfg_adds_conservative_memory_edges_for_repeated_array_accesses(self) -> None:
        function = Function(
            name="memory_chain",
            params=[Parameter("A", "i32[]"), Parameter("x", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        StoreOp("A", 0, "x"),
                        LoadOp("y", "i32", "A", 0),
                    ],
                    terminator=ReturnOp("y"),
                )
            ],
        )

        info = build_dfg(function)
        graph = info.blocks["entry"]
        edge_set = {(edge.source, edge.target, edge.kind, edge.label) for edge in graph.edges}

        self.assertIn(("entry:0", "entry:1", "memory", "A"), edge_set)
        self.assertIn(("entry:1", "entry:term", "value", "y"), edge_set)
        self.assertNotIn(("entry:1", "entry:term", "sink", ""), edge_set)

    def test_dfg_pass_stores_analysis_in_pass_context(self) -> None:
        function = Function(
            name="chain",
            params=[Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[BinaryOp("add", "t0", "i32", "a", "b")],
                    terminator=ReturnOp("t0"),
                )
            ],
        )

        result = PassManager([dfg_pass()]).run_with_context(function)

        self.assertEqual(result.context.history, ["dfg"])
        self.assertIn("dfg", result.context.analyses)
        self.assertIn("entry", result.context.analyses["dfg"].blocks)

    def test_to_dot_renders_basic_block_dfg(self) -> None:
        function = Function(
            name="chain",
            params=[Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        BinaryOp("add", "t0", "i32", "a", "b"),
                        BinaryOp("mul", "t1", "i32", "t0", 2),
                    ],
                    terminator=ReturnOp("t1"),
                )
            ],
        )

        graph = build_dfg(function).blocks["entry"]
        dot = to_dot(graph)

        self.assertIn('digraph "chain.entry.dfg"', dot)
        self.assertIn("node [shape=ellipse];", dot)
        self.assertIn('"entry:0" [label="t0:i32 = add a, b"];', dot)
        self.assertIn('"entry:0" -> "entry:1";', dot)
        self.assertIn('"entry:1" -> "entry:term";', dot)
        self.assertNotIn('color="#9c9c9c", style=dashed', dot)

    def test_compact_dfg_dot_uses_operator_only_nodes_and_labeled_edges(self) -> None:
        function = Function(
            name="chain",
            params=[Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        BinaryOp("add", "t0", "i32", "a", "b"),
                        BinaryOp("mul", "t1", "i32", "t0", 2),
                    ],
                    terminator=ReturnOp("t1"),
                )
            ],
        )

        dot = to_basic_block_dfg_dot(build_dfg(function).blocks["entry"], compact=True)

        self.assertIn('"entry:0" [label="+"];', dot)
        self.assertIn('"entry:1" [label="*"];', dot)
        self.assertIn('"entry:term" [label="ret"];', dot)
        self.assertIn(
            '"entry:in:a" [label="a", shape=box, style=filled, fillcolor="#eeeeee", color="#bdbdbd"];',
            dot,
        )
        self.assertIn(
            '"entry:in:b" [label="b", shape=box, style=filled, fillcolor="#eeeeee", color="#bdbdbd"];',
            dot,
        )
        self.assertIn('"entry:in:a" -> "entry:0" [label="a"];', dot)
        self.assertIn('"entry:in:b" -> "entry:0" [label="b"];', dot)
        self.assertIn('"entry:0" -> "entry:1" [label="t0"];', dot)
        self.assertIn('"entry:1" -> "entry:term" [label="t1"];', dot)
        self.assertNotIn("add a, b", dot)

    def test_to_cdfg_dot_renders_cfg_edges_between_block_dfg_clusters(self) -> None:
        function = Function(
            name="select",
            params=[Parameter("sel", "i1"), Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block("entry", terminator=CondBranchOp("sel", "then_blk", "else_blk")),
                Block(
                    "then_blk",
                    instructions=[BinaryOp("add", "x", "i32", "a", 1)],
                    terminator=ReturnOp("x"),
                ),
                Block(
                    "else_blk",
                    instructions=[BinaryOp("sub", "y", "i32", "b", 1)],
                    terminator=ReturnOp("y"),
                ),
            ],
        )

        dot = to_cdfg_dot(function)

        self.assertIn('digraph "select.cdfg"', dot)
        self.assertIn('subgraph "cluster_entry"', dot)
        self.assertIn('subgraph "cluster_then_blk"', dot)
        self.assertIn('subgraph "cluster_else_blk"', dot)
        self.assertIn("node [shape=ellipse];", dot)
        self.assertIn('"then_blk:0" -> "then_blk:term";', dot)
        self.assertIn('"else_blk:0" -> "else_blk:term";', dot)
        self.assertNotIn('"then_blk:0" -> "then_blk:term" [color="#9c9c9c", style=dashed];', dot)
        self.assertNotIn('"else_blk:0" -> "else_blk:term" [color="#9c9c9c", style=dashed];', dot)
        self.assertIn('ltail="cluster_entry", lhead="cluster_then_blk"', dot)
        self.assertIn('ltail="cluster_entry", lhead="cluster_else_blk"', dot)

    def test_compact_cdfg_dot_uses_compact_labels_inside_block_dfgs(self) -> None:
        function = Function(
            name="select",
            params=[Parameter("sel", "i1"), Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block("entry", terminator=CondBranchOp("sel", "then_blk", "else_blk")),
                Block(
                    "then_blk",
                    instructions=[BinaryOp("add", "x", "i32", "a", 1)],
                    terminator=ReturnOp("x"),
                ),
                Block(
                    "else_blk",
                    instructions=[BinaryOp("sub", "y", "i32", "b", 1)],
                    terminator=ReturnOp("y"),
                ),
            ],
        )

        dot = to_cdfg_dot(function, compact=True)

        self.assertIn(
            '"entry:in:sel" [label="sel", shape=box, style=filled, fillcolor="#eeeeee", color="#bdbdbd"];',
            dot,
        )
        self.assertIn('"entry:in:sel" -> "entry:term" [label="sel"];', dot)
        self.assertIn(
            '"then_blk:in:a" [label="a", shape=box, style=filled, fillcolor="#eeeeee", color="#bdbdbd"];',
            dot,
        )
        self.assertIn('"then_blk:in:a" -> "then_blk:0" [label="a"];', dot)
        self.assertIn(
            '"else_blk:in:b" [label="b", shape=box, style=filled, fillcolor="#eeeeee", color="#bdbdbd"];',
            dot,
        )
        self.assertIn('"else_blk:in:b" -> "else_blk:0" [label="b"];', dot)
        self.assertIn('"then_blk:0" [label="+"];', dot)
        self.assertIn('"else_blk:0" [label="-"];', dot)
        self.assertIn('"then_blk:0" -> "then_blk:term" [label="x"];', dot)
        self.assertIn('"else_blk:0" -> "else_blk:term" [label="y"];', dot)
        self.assertNotIn("add a, 1", dot)

    def test_compact_dfg_dot_shows_phi_incoming_values_as_block_inputs(self) -> None:
        function = Function(
            name="loop_header",
            params=[],
            return_type="i32",
            blocks=[
                Block(
                    "header",
                    instructions=[PhiOp("i", "i32", [("entry", "i0"), ("body", "i1")])],
                    terminator=ReturnOp("i"),
                )
            ],
        )

        dot = to_basic_block_dfg_dot(build_dfg(function).blocks["header"], compact=True)

        self.assertIn(
            '"header:in:i0" [label="i0", shape=box, style=filled, fillcolor="#eeeeee", color="#bdbdbd"];',
            dot,
        )
        self.assertIn(
            '"header:in:i1" [label="i1", shape=box, style=filled, fillcolor="#eeeeee", color="#bdbdbd"];',
            dot,
        )
        self.assertIn('"header:in:i0" -> "header:0" [label="i0"];', dot)
        self.assertIn('"header:in:i1" -> "header:0" [label="i1"];', dot)


class OptPassTests(unittest.TestCase):
    """Behavioral coverage for optimization and transform passes."""

    def test_const_prop_function_folds_constant_arithmetic_and_return_operands(self) -> None:
        function = Function(
            name="const_fold",
            params=[],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        ConstOp("a", "i32", 7),
                        BinaryOp("add", "b", "i32", "a", 1),
                        BinaryOp("mul", "c", "i32", "b", 2),
                    ],
                    terminator=ReturnOp("c"),
                )
            ],
        )

        result = const_prop_function(function)
        verify_function(result)

        self.assertTrue(all(isinstance(instruction, ConstOp) for instruction in result.blocks[0].instructions))
        self.assertIsInstance(result.blocks[0].terminator.value, Literal)
        self.assertEqual(result.blocks[0].terminator.value.value, 16)
        self.assertEqual(run_uir(result, {}).return_value, 16)

    def test_const_prop_function_folds_constant_branches_and_phi_values(self) -> None:
        function = Function(
            name="const_branch",
            params=[],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[ConstOp("cond", "i1", 1)],
                    terminator=CondBranchOp("cond", "then_blk", "else_blk"),
                ),
                Block("then_blk", instructions=[ConstOp("x_then", "i32", 5)], terminator=BranchOp("merge")),
                Block("else_blk", instructions=[ConstOp("x_else", "i32", 5)], terminator=BranchOp("merge")),
                Block(
                    "merge",
                    instructions=[PhiOp("x", "i32", [("then_blk", "x_then"), ("else_blk", "x_else")])],
                    terminator=ReturnOp("x"),
                ),
            ],
        )

        result = const_prop_function(function)
        verify_function(result)

        self.assertIsInstance(result.blocks[0].terminator, BranchOp)
        self.assertEqual(result.blocks[0].terminator.target, "then_blk")
        self.assertIsInstance(result.blocks[-1].instructions[0], ConstOp)
        self.assertEqual(result.blocks[-1].instructions[0].value, 5)
        self.assertEqual(run_uir(result, {}).return_value, 5)

    def test_simplify_cfg_function_prunes_and_simplifies_cfg(self) -> None:
        function = Function(
            name="cleanup",
            params=[],
            return_type="i32",
            blocks=[
                Block("entry", terminator=BranchOp("jump")),
                Block("jump", terminator=BranchOp("work")),
                Block(
                    "work",
                    instructions=[ConstOp("x", "i32", 7)],
                    terminator=CondBranchOp(1, "exit", "exit"),
                ),
                Block("exit", terminator=ReturnOp("x")),
                Block("dead", instructions=[ConstOp("y", "i32", 9)], terminator=ReturnOp("y")),
            ],
        )

        result = simplify_cfg_function(function)
        verify_function(result)

        self.assertNotIn("dead", [block.label for block in result.blocks])
        self.assertNotIn("jump", [block.label for block in result.blocks])
        self.assertEqual(run_uir(result, []).return_value, 7)

    def test_inline_calls_clones_callee_cfg_into_caller(self) -> None:
        callee = Function(
            name="select_add",
            params=[Parameter("sel", "i1"), Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block("entry", terminator=CondBranchOp("sel", "then_blk", "else_blk")),
                Block(
                    "then_blk",
                    instructions=[BinaryOp("add", "x", "i32", "a", 1)],
                    terminator=ReturnOp("x"),
                ),
                Block(
                    "else_blk",
                    instructions=[BinaryOp("add", "x", "i32", "b", 2)],
                    terminator=ReturnOp("x"),
                ),
            ],
        )
        caller = Function(
            name="caller",
            params=[Parameter("sel", "i1"), Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        CallOp("select_add", ["sel", "a", "b"], dest="tmp", type="i32"),
                        BinaryOp("mul", "y", "i32", "tmp", 2),
                    ],
                    terminator=ReturnOp("y"),
                )
            ],
        )
        module = Module(functions=[callee, caller])

        inlined = inline_calls(module)
        verify_module(inlined, allow_calls=False)
        inlined_caller = inlined.get_function("caller")
        assert inlined_caller is not None

        self.assertFalse(
            any(isinstance(instruction, CallOp) for block in inlined_caller.blocks for instruction in block.instructions)
        )
        self.assertEqual(run_uir(inlined_caller, {"sel": 1, "a": 7, "b": 100}).return_value, 16)
        self.assertEqual(run_uir(inlined_caller, {"sel": 0, "a": 7, "b": 100}).return_value, 204)

    def test_inline_calls_can_restrict_inlining_to_selected_callees(self) -> None:
        keep = Function(
            name="keep",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
        )
        inline_me = Function(
            name="inline_me",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[Block("entry", instructions=[BinaryOp("mul", "y", "i32", "x", 2)], terminator=ReturnOp("y"))],
        )
        caller = Function(
            name="caller",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        CallOp("keep", ["x"], dest="a", type="i32"),
                        CallOp("inline_me", ["a"], dest="b", type="i32"),
                    ],
                    terminator=ReturnOp("b"),
                )
            ],
        )
        module = Module(functions=[keep, inline_me, caller])

        inlined = inline_calls(module, pass_args=("inline_me",))
        verify_module(inlined)
        inlined_caller = inlined.get_function("caller")
        assert inlined_caller is not None

        remaining_calls = [
            instruction.callee
            for block in inlined_caller.blocks
            for instruction in block.instructions
            if isinstance(instruction, CallOp)
        ]
        self.assertEqual(remaining_calls, ["keep"])
        self.assertEqual(run_uir(inlined_caller, {"x": 5}, module=inlined).return_value, 12)

    def test_dce_function_removes_dead_scalar_instructions_but_keeps_side_effects(self) -> None:
        function = Function(
            name="cleanup",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        BinaryOp("add", "dead0", "i32", "x", 1),
                        BinaryOp("mul", "live0", "i32", "x", 2),
                        PrintOp("x=%d", ["dead0"]),
                        BinaryOp("add", "dead1", "i32", "live0", 5),
                    ],
                    terminator=ReturnOp("live0"),
                )
            ],
        )

        result = dce_function(function)
        verify_function(result)

        kept_opcodes = [instruction.opcode for instruction in result.blocks[0].instructions]
        self.assertEqual(kept_opcodes, ["add", "mul", "print"])
        self.assertEqual(run_uir(result, {"x": 7}).return_value, 14)

    def test_prune_functions_module_removes_callees_made_dead_by_inlining(self) -> None:
        callee = Function(
            name="helper",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
        )
        caller = Function(
            name="worker",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[Block("entry", instructions=[CallOp("helper", ["x"], dest="y", type="i32")], terminator=ReturnOp("y"))],
        )
        main = Function(
            name="main",
            params=[],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[CallOp("worker", [7], dest="z", type="i32")],
                    terminator=ReturnOp("z"),
                )
            ],
        )
        module = Module(functions=[callee, caller, main])

        inlined = inline_calls(module)
        pruned = prune_functions_module(inlined)
        verify_module(pruned)

        self.assertIsNotNone(pruned.get_function("main"))
        self.assertIsNotNone(pruned.get_function("worker"))
        self.assertIsNone(pruned.get_function("helper"))

    def test_cse_function_eliminates_duplicate_expressions_and_rewrites_successor_uses(self) -> None:
        function = Function(
            name="cleanup",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        BinaryOp("add", "a0", "i32", "x", 1),
                        BinaryOp("add", "a1", "i32", "x", 1),
                    ],
                    terminator=BranchOp("exit"),
                ),
                Block("exit", instructions=[BinaryOp("mul", "y", "i32", "a1", 2)], terminator=ReturnOp("y")),
            ],
        )

        result = cse_function(function)
        verify_function(result)

        entry_opcodes = [instruction.opcode for instruction in result.blocks[0].instructions]
        exit_texts = [instruction for instruction in result.blocks[1].instructions]

        self.assertEqual(entry_opcodes, ["add"])
        self.assertEqual(exit_texts[0].lhs, "a0")
        self.assertEqual(run_uir(result, {"x": 7}).return_value, 16)

    def test_copy_prop_function_rewrites_successor_uses_of_mov_copies(self) -> None:
        function = Function(
            name="cleanup",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    "entry",
                    instructions=[
                        BinaryOp("add", "a0", "i32", "x", 1),
                        UnaryOp("mov", "a1", "i32", "a0"),
                    ],
                    terminator=BranchOp("exit"),
                ),
                Block("exit", instructions=[BinaryOp("mul", "y", "i32", "a1", 2)], terminator=ReturnOp("y")),
            ],
        )

        result = copy_prop_function(function)
        verify_function(result)

        exit_instruction = result.blocks[1].instructions[0]
        self.assertEqual(exit_instruction.lhs, "a0")
        self.assertEqual(run_uir(result, {"x": 7}).return_value, 16)

    def test_copy_prop_function_rewrites_phi_incoming_values_from_mov_copies(self) -> None:
        function = Function(
            name="phi_cleanup",
            params=[Parameter("sel", "i1"), Parameter("a", "i32"), Parameter("b", "i32")],
            return_type="i32",
            blocks=[
                Block("entry", instructions=[], terminator=CondBranchOp("sel", "then_blk", "else_blk")),
                Block(
                    "then_blk",
                    instructions=[
                        BinaryOp("add", "t0", "i32", "a", 1),
                        UnaryOp("mov", "x_then", "i32", "t0"),
                    ],
                    terminator=BranchOp("merge"),
                ),
                Block(
                    "else_blk",
                    instructions=[BinaryOp("sub", "x_else", "i32", "b", 1)],
                    terminator=BranchOp("merge"),
                ),
                Block(
                    "merge",
                    instructions=[PhiOp("x", "i32", [("then_blk", "x_then"), ("else_blk", "x_else")])],
                    terminator=ReturnOp("x"),
                ),
            ],
        )

        result = copy_prop_function(function)
        verify_function(result)

        merge_phi = result.block_map()["merge"].instructions[0]
        self.assertEqual(merge_phi.incoming[0].value, "t0")
        self.assertEqual(merge_phi.incoming[1].value, "x_else")
        self.assertEqual(run_uir(result, {"sel": 1, "a": 7, "b": 20}).return_value, 8)
        self.assertEqual(run_uir(result, {"sel": 0, "a": 7, "b": 20}).return_value, 19)
