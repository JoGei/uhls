from __future__ import annotations

import unittest

from uhls.backend.hls.uhir import build_sequencing_graph, format_uhir, lower_module_to_seq, parse_uhir, to_dot
from uhls.frontend import lower_source_to_uir
from uhls.middleend.uir import BinaryOp, Block, CallOp, Function, Module, Parameter, ReturnOp, parse_module


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

    def test_lower_frontend_for_loop_keeps_branch_based_shape_before_gopt_loop_dialect(self) -> None:
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
        self.assertNotIn("loop", [node.opcode for node in proc.nodes])
        self.assertIn("branch", [node.opcode for node in proc.nodes])
        loop_region = design.get_region("loop_sum4_for_header_1")
        self.assertIsNone(loop_region)
        self.assertIsNone(design.get_region("bb_sum4_entry"))
        self.assertIsNone(design.get_region("bb_sum4_for_exit_4"))
        self.assertIsNone(design.get_region("bb_sum4_for_header_1"))
        self.assertIsNone(design.get_region("bb_sum4_for_body_2"))
        self.assertIn("phi", [node.opcode for node in proc.nodes])
        self.assertIn("lt", [node.opcode for node in proc.nodes])

        body_region = next((region for region in design.regions if region.id.startswith("loop_body_")), None)
        empty_region = next((region for region in design.regions if region.kind == "empty"), None)
        self.assertIsNotNone(body_region)
        self.assertIsNotNone(empty_region)
        assert body_region is not None
        assert empty_region is not None
        self.assertEqual(body_region.parent, proc.id)
        self.assertEqual(empty_region.parent, proc.id)
        self.assertEqual(body_region.kind, "basicblock")
        self.assertIn("load", [node.opcode for node in body_region.nodes])
        self.assertIn("add", [node.opcode for node in body_region.nodes])
        body_add = next(node for node in body_region.nodes if node.opcode == "add")
        body_source = next(node for node in body_region.nodes if node.opcode == "nop" and node.attributes.get("role") == "source")
        body_edges = {(edge.kind, edge.source, edge.target) for edge in body_region.edges}
        self.assertNotIn(("data", body_source.id, body_add.id), body_edges)

        proc_branch = next(node for node in proc.nodes if node.opcode == "branch")
        self.assertNotIn("static_trip_count", proc_branch.attributes)
        proc_seq_edges = {(edge.source, edge.target) for edge in proc.edges if edge.kind == "seq"}
        proc_data_edges = {(edge.source, edge.target) for edge in proc.edges if edge.kind == "data"}
        self.assertIn((proc_branch.id, body_region.id), proc_seq_edges)
        self.assertIn((body_region.id, proc_branch.id), proc_seq_edges)
        self.assertIn((proc_branch.id, next(node for node in proc.nodes if node.opcode == "ret").id), proc_data_edges)
        self.assertNotIn("bb_sum4_entry", [ref.target for ref in proc.region_refs])
        self.assertNotIn("bb_sum4_for_exit_4", [ref.target for ref in proc.region_refs])

        self.assertNotIn("static_trip_count", proc_branch.attributes)
        self.assertEqual(proc_branch.attributes.get("true_child"), body_region.id)
        self.assertEqual(proc_branch.attributes.get("false_child"), empty_region.id)
        self.assertIn((body_region.id, proc_branch.id), proc_seq_edges)
        self.assertIn((empty_region.id, proc_branch.id), proc_seq_edges)
        self.assertIn((next(node for node in proc.nodes if node.opcode == "lt").id, proc_branch.id), proc_data_edges)

    def test_lower_frontend_runtime_bound_loop_has_no_static_trip_count(self) -> None:
        module = lower_source_to_uir(
            """
            int32_t sumN(int32_t A[4], int32_t N) {
                int32_t i;
                int32_t sum = 0;
                for (i = 0; i < N; i = i + 1) {
                    sum = sum + A[i];
                }
                return sum;
            }
            """
        )

        design = lower_module_to_seq(module, top="sumN")
        proc = design.get_region("proc_sumN")
        self.assertIsNotNone(proc)
        assert proc is not None
        proc_branch = next(node for node in proc.nodes if node.opcode == "branch")
        self.assertNotIn("static_trip_count", proc_branch.attributes)

    def test_lower_parsed_uir_loop_has_no_static_trip_count(self) -> None:
        module = parse_module(
            """
            func dot4(A:i32[], B:i32[]) -> i32

            block entry:
                br for_header_1


            block for_header_1:
                i_1:i32 = phi(entry: 0:i32, for_body_2: t4_0)
                sum_1:i32 = phi(entry: 0:i32, for_body_2: inl_mac_0_t1_0)
                t0_0:i1 = lt i_1, 4:i32
                cbr t0_0, for_body_2, for_exit_4


            block for_body_2:
                t1_0:i32 = load A[i_1]
                t2_0:i32 = load B[i_1]
                inl_mac_0_t0_0:i32 = mul t1_0, t2_0
                inl_mac_0_t1_0:i32 = add sum_1, inl_mac_0_t0_0
                t4_0:i32 = add i_1, 1:i32
                br for_header_1


            block for_exit_4:
                ret sum_1
            """
        )

        design = lower_module_to_seq(module, top="dot4")
        proc = design.get_region("proc_dot4")
        self.assertIsNotNone(proc)
        assert proc is not None
        proc_branch = next(node for node in proc.nodes if node.opcode == "branch")
        self.assertNotIn("static_trip_count", proc_branch.attributes)

    def test_lower_parsed_uir_post_loop_compare_depends_on_loop_helper(self) -> None:
        module = parse_module(
            """
            func dot4(A:i32[], B:i32[]) -> i32

            block entry:
                br for_header_1


            block for_header_1:
                i_1:i32 = phi(entry: 0:i32, for_body_2: t4_0)
                sum_1:i32 = phi(entry: 0:i32, for_body_2: inl_mac_0_t1_0)
                t0_0:i1 = lt i_1, 4:i32
                cbr t0_0, for_body_2, for_exit_4


            block for_body_2:
                t1_0:i32 = load A[i_1]
                t2_0:i32 = load B[i_1]
                inl_mac_0_t0_0:i32 = mul t1_0, t2_0
                inl_mac_0_t1_0:i32 = add sum_1, inl_mac_0_t0_0
                t4_0:i32 = add i_1, 1:i32
                br for_header_1


            block for_exit_4:
                t5_0:i1 = lt sum_1, 0:i32
                cbr t5_0, if_then_5, if_end_7


            block if_then_5:
                br if_end_7


            block if_end_7:
                sum_4:i32 = phi(for_exit_4: sum_1, if_then_5: 0:i32)
                ret sum_4
            """
        )

        design = lower_module_to_seq(module, top="dot4")
        proc = design.get_region("proc_dot4")
        self.assertIsNotNone(proc)
        assert proc is not None

        compare_node = next(node for node in proc.nodes if node.opcode == "lt" and node.operands == ("sum_1", "0:i32"))
        loop_branch_node = next(node for node in proc.nodes if node.opcode == "branch" and node.operands == ("t0_0",))
        branch_node = next(node for node in proc.nodes if node.opcode == "branch" and node.operands == ("t5_0",))
        phi_node = next(node for node in proc.nodes if node.opcode == "phi" and node.operands == ("sum_1", "0:i32"))
        ret_node = next(node for node in proc.nodes if node.opcode == "ret")
        proc_data_edges = {(edge.source, edge.target) for edge in proc.edges if edge.kind == "data"}

        self.assertIn((loop_branch_node.id, compare_node.id), proc_data_edges)
        self.assertIn((branch_node.id, phi_node.id), proc_data_edges)
        self.assertIn((phi_node.id, ret_node.id), proc_data_edges)
        self.assertNotIn((branch_node.id, compare_node.id), proc_data_edges)
        self.assertIsNone(design.get_region("bb_dot4_if_end_7"))

        then_region = design.get_region("bb_dot4_if_then_5")
        self.assertIsNotNone(then_region)
        assert then_region is not None
        then_source = next(node for node in then_region.nodes if node.opcode == "nop" and node.attributes.get("role") == "source")
        then_sink = next(node for node in then_region.nodes if node.opcode == "nop" and node.attributes.get("role") == "sink")
        then_data_edges = {(edge.source, edge.target) for edge in then_region.edges if edge.kind == "data"}
        self.assertIn((then_source.id, then_sink.id), then_data_edges)

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
