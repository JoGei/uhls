from __future__ import annotations

import unittest

from uhls.backend.uhir import build_sequencing_graph, format_uhir, lower_module_to_seq, parse_uhir, to_dot
from uhls.frontend import lower_source_to_uir
from uhls.middleend.uir import BinaryOp, Block, CallOp, Function, Module, Parameter, ReturnOp


class SequencingGraphLoweringTests(unittest.TestCase):
    """Coverage for µIR to seq-stage µhIR lowering."""

    def test_lower_simple_function_builds_polar_procedure_and_block_units(self) -> None:
        module = Module(
            name="demo",
            functions=[
                Function(
                    name="add1",
                    params=[Parameter("x", "i32")],
                    return_type="i32",
                    blocks=[
                        Block(
                            "entry",
                            instructions=[BinaryOp("add", "y", "i32", "x", 1)],
                            terminator=ReturnOp("y"),
                        )
                    ],
                )
            ],
        )

        design = lower_module_to_seq(module)
        rendered = format_uhir(design)
        reparsed = parse_uhir(rendered)

        self.assertEqual(design.stage, "seq")
        self.assertEqual(design.name, "demo")
        self.assertEqual(design.inputs[0].name, "x")
        self.assertEqual(design.outputs[0].name, "result")
        self.assertEqual(reparsed.stage, "seq")

        proc = design.get_region("proc_add1")
        self.assertIsNotNone(proc)
        assert proc is not None
        self.assertEqual(proc.kind, "procedure")
        self.assertEqual([node.opcode for node in proc.nodes[:2]], ["nop", "nop"])
        self.assertIsNone(design.get_region("bb_add1_entry"))
        self.assertIn("add", [node.opcode for node in proc.nodes])
        self.assertIn("ret", [node.opcode for node in proc.nodes])

    def test_lower_calls_create_call_helper_vertices_and_callee_units(self) -> None:
        callee = Function(
            name="callee",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
        )
        caller = Function(
            name="caller",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[Block("entry", instructions=[CallOp("callee", ["x"], dest="z", type="i32")], terminator=ReturnOp("z"))],
        )
        design = lower_module_to_seq(
            Module(functions=[callee, caller]),
            top="caller",
        )

        proc = design.get_region("proc_caller")
        self.assertIsNotNone(proc)
        assert proc is not None
        self.assertIn("proc_callee", [ref.target for ref in proc.region_refs])
        self.assertIn("call", [node.opcode for node in proc.nodes])
        self.assertIsNotNone(design.get_region("proc_callee"))

        call_node = next(node for node in proc.nodes if node.opcode == "call")
        ret_node = next(node for node in proc.nodes if node.opcode == "ret")
        seq_edges = [edge for edge in proc.edges if edge.kind == "seq"]
        data_edges = [edge for edge in proc.edges if edge.kind == "data"]
        self.assertIn((call_node.id, "proc_callee"), [(edge.source, edge.target) for edge in seq_edges])
        self.assertIn(("proc_callee", call_node.id), [(edge.source, edge.target) for edge in seq_edges])
        self.assertIn((call_node.id, ret_node.id), [(edge.source, edge.target) for edge in data_edges])
        self.assertTrue(
            next(edge for edge in seq_edges if edge.source == call_node.id and edge.target == "proc_callee").attributes["hierarchy"]
        )
        self.assertTrue(
            next(edge for edge in seq_edges if edge.source == "proc_callee" and edge.target == call_node.id).attributes["hierarchy"]
        )

    def test_build_sequencing_graph_exposes_internal_sgu_hierarchy(self) -> None:
        module = Module(
            name="demo",
            functions=[
                Function(
                    name="add1",
                    params=[Parameter("x", "i32")],
                    return_type="i32",
                    blocks=[
                        Block(
                            "entry",
                            instructions=[BinaryOp("add", "y", "i32", "x", 1)],
                            terminator=ReturnOp("y"),
                        )
                    ],
                )
            ],
        )

        graph = build_sequencing_graph(module)
        proc = graph.get_unit("proc_add1")
        self.assertIsNotNone(proc)
        assert proc is not None
        self.assertEqual(proc.kind, "procedure")
        self.assertEqual(graph.name, "demo")
        self.assertEqual(proc.region_refs, [])
        self.assertEqual([edge.kind for edge in proc.edges].count("seq"), 0)

    def test_lower_frontend_if_creates_branch_helper_vertices(self) -> None:
        module = lower_source_to_uir(
            """
            int32_t select(int32_t sel, int32_t a, int32_t b) {
                int32_t x;
                if (sel < 0) {
                    x = a + 1;
                } else {
                    x = b - 1;
                }
                return x;
            }
            """
        )

        design = lower_module_to_seq(module, top="select")
        proc = design.get_region("proc_select")
        self.assertIsNotNone(proc)
        assert proc is not None
        self.assertIn("branch", [node.opcode for node in proc.nodes])

    def test_lower_frontend_for_loop_creates_loop_helper_and_loop_region(self) -> None:
        module = lower_source_to_uir(
            """
            int32_t sum4(int32_t A[4]) {
                int32_t i;
                int32_t sum = 0;
                for (i = 0; i < 4; i = i + 1) {
                    sum = sum + A[i];
                }
                return sum;
            }
            """
        )

        design = lower_module_to_seq(module, top="sum4")
        proc = design.get_region("proc_sum4")
        self.assertIsNotNone(proc)
        assert proc is not None
        self.assertIn("loop", [node.opcode for node in proc.nodes])
        loop_region = next((region for region in design.regions if region.kind == "loop"), None)
        self.assertIsNotNone(loop_region)
        assert loop_region is not None
        self.assertIn("branch", [node.opcode for node in loop_region.nodes])
        self.assertIsNone(design.get_region("bb_sum4_entry"))
        self.assertIsNone(design.get_region("bb_sum4_for_exit_4"))
        self.assertIsNone(design.get_region("bb_sum4_for_header_1"))
        self.assertIsNone(design.get_region("bb_sum4_for_body_2"))
        self.assertIn("phi", [node.opcode for node in loop_region.nodes])
        self.assertIn("lt", [node.opcode for node in loop_region.nodes])

        body_region = next((region for region in design.regions if region.kind == "body"), None)
        empty_region = next((region for region in design.regions if region.kind == "empty"), None)
        self.assertIsNotNone(body_region)
        self.assertIsNotNone(empty_region)
        assert body_region is not None
        assert empty_region is not None
        self.assertEqual(body_region.parent, loop_region.id)
        self.assertEqual(empty_region.parent, loop_region.id)
        self.assertIn("load", [node.opcode for node in body_region.nodes])
        self.assertIn("add", [node.opcode for node in body_region.nodes])
        body_add = next(node for node in body_region.nodes if node.opcode == "add")
        body_source = next(node for node in body_region.nodes if node.opcode == "nop" and node.attributes.get("role") == "source")
        body_edges = {(edge.kind, edge.source, edge.target) for edge in body_region.edges}
        self.assertNotIn(("data", body_source.id, body_add.id), body_edges)

        proc_loop = next(node for node in proc.nodes if node.opcode == "loop")
        proc_seq_edges = {(edge.source, edge.target) for edge in proc.edges if edge.kind == "seq"}
        proc_data_edges = {(edge.source, edge.target) for edge in proc.edges if edge.kind == "data"}
        self.assertIn((proc_loop.id, loop_region.id), proc_seq_edges)
        self.assertIn((loop_region.id, proc_loop.id), proc_seq_edges)
        self.assertIn((proc_loop.id, next(node for node in proc.nodes if node.opcode == "ret").id), proc_data_edges)
        self.assertNotIn("bb_sum4_entry", [ref.target for ref in proc.region_refs])
        self.assertNotIn("bb_sum4_for_exit_4", [ref.target for ref in proc.region_refs])

        loop_branch = next(node for node in loop_region.nodes if node.opcode == "branch")
        loop_seq_edges = {(edge.source, edge.target) for edge in loop_region.edges if edge.kind == "seq"}
        loop_data_edges = {(edge.source, edge.target) for edge in loop_region.edges if edge.kind == "data"}
        self.assertIn((body_region.id, loop_branch.id), loop_seq_edges)
        self.assertIn((empty_region.id, loop_branch.id), loop_seq_edges)
        self.assertIn((next(node for node in loop_region.nodes if node.opcode == "lt").id, loop_branch.id), loop_data_edges)

    def test_dot_renders_hierarchy_edges_as_dashed(self) -> None:
        callee = Function(
            name="callee",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
        )
        caller = Function(
            name="caller",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[Block("entry", instructions=[CallOp("callee", ["x"], dest="z", type="i32")], terminator=ReturnOp("z"))],
        )

        design = lower_module_to_seq(
            Module(functions=[callee, caller]),
            top="caller",
        )
        dot = to_dot(design)

        self.assertIn('style=dashed', dot)
        self.assertIn('color="#4c78a8"', dot)
        self.assertIn('color="#222222"', dot)
