from __future__ import annotations

import unittest

from uhls.backend.uhir import (
    ExecutabilityGraph,
    dummy_executability_graph,
    executability_graph_from_uhir,
    format_uhir,
    lower_module_to_seq,
    lower_seq_to_alloc,
    parse_uhir,
)
from uhls.backend.hls.alloc import executability_graph_to_dot, format_executability_graph
from uhls.middleend.uir import COMPACT_OPCODE_LABELS, BinaryOp, Block, CallOp, Function, Module, Parameter, ReturnOp


def _full_executability_graph() -> ExecutabilityGraph:
    operations = tuple(sorted(COMPACT_OPCODE_LABELS))
    weighted_edges = [("fu_generic", operation, 1, 1) for operation in operations]
    weighted_edges = [edge for edge in weighted_edges if not (edge[0] == "fu_generic" and edge[1] == "add")]
    weighted_edges.append(("fu_generic", "add", 2, 2))
    weighted_edges.append(("fu_fast_add", "add", 1, 1))
    return ExecutabilityGraph(
        functional_units=("fu_generic", "fu_fast_add"),
        operations=operations,
        edges=tuple(weighted_edges),
    )


class AllocationLoweringTests(unittest.TestCase):
    """Coverage for seq-to-alloc lowering."""

    def test_dummy_executability_graph_covers_all_ops_with_ewms_delay_one(self) -> None:
        graph = dummy_executability_graph()

        self.assertEqual(graph.functional_units, ("EWMS",))
        self.assertEqual(set(graph.operations), set(COMPACT_OPCODE_LABELS))
        self.assertIn(("EWMS", "add", 1, 1), graph.edges)
        self.assertTrue(all(edge[0] == "EWMS" and edge[2] == 1 and edge[3] == 1 for edge in graph.edges))

    def test_executability_graph_render_helpers_include_weights(self) -> None:
        graph = ExecutabilityGraph(
            functional_units=("fu_generic",),
            operations=("add",),
            edges=(("fu_generic", "add", 1, 1),),
        )

        rendered = format_executability_graph(graph)
        dot = executability_graph_to_dot(graph)

        self.assertIn("executability_graph {", rendered)
        self.assertIn("edge FU_GENERIC -- add ii=1 d=1", rendered)
        self.assertIn('graph "executability_graph"', dot)
        self.assertIn('label="ii=1, d=1"', dot)

    def test_executability_graph_can_be_loaded_from_exg_uhir(self) -> None:
        graph = executability_graph_from_uhir(
            parse_uhir(
                """
                design demo_exg
                stage exg

                region G0 kind=executability {
                  node FU_GENERIC = fu partition=fu
                  node add = op partition=op
                  edge exg FU_GENERIC -- add ii=1 d=1
                }
                """
            )
        )

        self.assertEqual(graph.functional_units, ("FU_GENERIC",))
        self.assertEqual(graph.operations, ("add",))
        self.assertEqual(graph.edges, (("FU_GENERIC", "add", 1, 1),))

    def test_lower_seq_to_alloc_annotates_nodes_with_selected_fu_and_delay(self) -> None:
        module = Module(
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

        seq_design = lower_module_to_seq(module)
        alloc_design = lower_seq_to_alloc(seq_design, executability_graph=_full_executability_graph())
        rendered = format_uhir(alloc_design)
        reparsed = parse_uhir(rendered)

        self.assertEqual(alloc_design.stage, "alloc")
        proc = alloc_design.get_region("proc_add1")
        self.assertIsNotNone(proc)
        assert proc is not None

        add_node = next(node for node in proc.nodes if node.opcode == "add")
        ret_node = next(node for node in proc.nodes if node.opcode == "ret")
        source_node = next(node for node in proc.nodes if node.opcode == "nop" and node.attributes.get("role") == "source")

        self.assertEqual(add_node.attributes["class"], "FU_FAST_ADD")
        self.assertEqual(add_node.attributes["ii"], 1)
        self.assertEqual(add_node.attributes["delay"], 1)
        self.assertEqual(ret_node.attributes["class"], "CTRL")
        self.assertEqual(ret_node.attributes["ii"], 0)
        self.assertEqual(ret_node.attributes["delay"], 0)
        self.assertEqual(source_node.attributes["class"], "CTRL")
        self.assertEqual(source_node.attributes["ii"], 0)
        self.assertEqual(source_node.attributes["delay"], 0)
        exg_region = alloc_design.get_region("executability_graph")
        self.assertIsNotNone(exg_region)
        assert exg_region is not None
        self.assertEqual(exg_region.kind, "executability")
        self.assertTrue(any((node.id == "FU_FAST_ADD" or node.attributes.get("name") == "FU_FAST_ADD") and node.opcode == "fu" for node in exg_region.nodes))
        self.assertTrue(any((node.id == "CTRL" or node.attributes.get("name") == "CTRL") and node.opcode == "fu" for node in exg_region.nodes))
        self.assertTrue(any(node.id == "add" and node.opcode == "op" for node in exg_region.nodes))
        self.assertTrue(any(node.id == "nop" and node.opcode == "op" for node in exg_region.nodes))
        self.assertTrue(any(not edge.directed and edge.kind == "exg" for edge in exg_region.edges))
        self.assertFalse(any(node.id == "sub" for node in exg_region.nodes))
        self.assertEqual(reparsed.stage, "alloc")

    def test_lower_seq_to_alloc_accepts_exg_uhir_collateral(self) -> None:
        module = Module(
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

        graph = parse_uhir(
            """
            design full_exg
            stage exg

            region G0 kind=executability {
              node FU_GENERIC = fu partition=fu
              node FU_FAST_ADD = fu partition=fu
            """
            + "\n".join(f"  node {operation} = op partition=op" for operation in sorted(COMPACT_OPCODE_LABELS))
            + "\n"
            + "\n".join(
                f"  edge exg FU_GENERIC -- {operation} ii=1 d=1"
                for operation in sorted(COMPACT_OPCODE_LABELS)
                if operation != "add"
            )
            + "\n"
            + "  edge exg FU_GENERIC -- add ii=2 d=2\n"
            + "  edge exg FU_FAST_ADD -- add ii=1 d=1\n"
            + "}\n"
        )

        alloc_design = lower_seq_to_alloc(lower_module_to_seq(module), executability_graph=graph)
        proc = alloc_design.get_region("proc_add1")
        self.assertIsNotNone(proc)
        assert proc is not None
        add_node = next(node for node in proc.nodes if node.opcode == "add")
        exg_region = alloc_design.get_region("executability_graph")

        self.assertEqual(add_node.attributes["class"], "FU_FAST_ADD")
        self.assertEqual(add_node.attributes["ii"], 1)
        self.assertEqual(add_node.attributes["delay"], 1)
        self.assertIsNotNone(exg_region)

    def test_lower_seq_to_alloc_can_select_min_ii_algorithm(self) -> None:
        seq_design = lower_module_to_seq(
            Module(
                functions=[
                    Function(
                        name="add1",
                        params=[Parameter("x", "i32")],
                        return_type="i32",
                        blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
                    )
                ],
            )
        )

        graph = ExecutabilityGraph(
            functional_units=("fu_fast_ii", "fu_fast_delay"),
            operations=tuple(sorted(COMPACT_OPCODE_LABELS)),
            edges=tuple(("fu_fast_delay", operation, 1, 1) for operation in sorted(COMPACT_OPCODE_LABELS) if operation != "add")
            + (("fu_fast_delay", "add", 2, 2), ("fu_fast_ii", "add", 1, 3)),
        )

        alloc_design = lower_seq_to_alloc(seq_design, executability_graph=graph, algorithm="min_ii")
        proc = alloc_design.get_region("proc_add1")
        self.assertIsNotNone(proc)
        assert proc is not None
        add_node = next(node for node in proc.nodes if node.opcode == "add")

        self.assertEqual(add_node.attributes["class"], "FU_FAST_II")
        self.assertEqual(add_node.attributes["ii"], 1)
        self.assertEqual(add_node.attributes["delay"], 3)

    def test_lower_seq_to_alloc_allocates_call_as_structural_ctrl_vertex(self) -> None:
        seq_design = lower_module_to_seq(
            Module(
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
        )

        alloc_design = lower_seq_to_alloc(seq_design, executability_graph=_full_executability_graph())
        proc = alloc_design.get_region("proc_caller")
        self.assertIsNotNone(proc)
        assert proc is not None
        call_node = next(node for node in proc.nodes if node.opcode == "call")

        self.assertEqual(call_node.attributes["class"], "CTRL")
        self.assertEqual(call_node.attributes["ii"], 0)
        self.assertEqual(call_node.attributes["delay"], 0)

    def test_lower_seq_to_alloc_rejects_unknown_algorithm(self) -> None:
        seq_design = lower_module_to_seq(
            Module(
                functions=[
                    Function(
                        name="add1",
                        params=[Parameter("x", "i32")],
                        return_type="i32",
                        blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
                    )
                ],
            )
        )

        with self.assertRaisesRegex(ValueError, "unsupported allocation algorithm"):
            lower_seq_to_alloc(seq_design, executability_graph=_full_executability_graph(), algorithm="mystery")

    def test_lower_seq_to_alloc_rejects_non_bipartite_graph(self) -> None:
        seq_design = lower_module_to_seq(
            Module(
                functions=[
                    Function(
                        name="add1",
                        params=[Parameter("x", "i32")],
                        return_type="i32",
                        blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
                    )
                ],
            )
        )

        graph = ExecutabilityGraph(
            functional_units=("fu_generic",),
            operations=tuple(sorted(COMPACT_OPCODE_LABELS)),
            edges=(("add", "sub", 1, 1),),
        )

        with self.assertRaisesRegex(ValueError, "not bipartite"):
            lower_seq_to_alloc(seq_design, executability_graph=graph)

    def test_lower_seq_to_alloc_requires_full_uir_coverage(self) -> None:
        seq_design = lower_module_to_seq(
            Module(
                functions=[
                    Function(
                        name="add1",
                        params=[Parameter("x", "i32")],
                        return_type="i32",
                        blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
                    )
                ],
            )
        )

        graph = ExecutabilityGraph(
            functional_units=("fu_generic",),
            operations=("add",),
            edges=(("fu_generic", "add", 1, 1),),
        )

        with self.assertRaisesRegex(ValueError, "does not cover all canonical"):
            lower_seq_to_alloc(seq_design, executability_graph=graph)

    def test_lower_seq_to_alloc_rejects_edge_weights_with_ii_greater_than_delay(self) -> None:
        seq_design = lower_module_to_seq(
            Module(
                functions=[
                    Function(
                        name="add1",
                        params=[Parameter("x", "i32")],
                        return_type="i32",
                        blocks=[Block("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
                    )
                ],
            )
        )

        graph = ExecutabilityGraph(
            functional_units=("fu_generic",),
            operations=tuple(sorted(COMPACT_OPCODE_LABELS)),
            edges=tuple(("fu_generic", operation, 1, 1) for operation in sorted(COMPACT_OPCODE_LABELS) if operation != "add")
            + (("fu_generic", "add", 2, 1),),
        )

        with self.assertRaisesRegex(ValueError, "ii<=d"):
            lower_seq_to_alloc(seq_design, executability_graph=graph)
