from __future__ import annotations

import unittest

from uhls.backend.hls.uhir import (
    ExecutabilityGraph,
    dummy_executability_graph,
    executability_graph_from_uhir,
    format_uhir,
    lower_module_to_seq,
    lower_seq_to_alloc,
    parse_uhir,
)
from uhls.backend.hls.alloc import executability_graph_to_dot, format_executability_graph
from uhls.backend.hls.impl import MemoryPolicy
from uhls.backend.hls.lib import validate_component_library
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


def _memory_executability_graph() -> ExecutabilityGraph:
    operations = tuple(sorted(COMPACT_OPCODE_LABELS))
    weighted_edges = [("fu_generic", operation, 1, 1) for operation in operations]
    weighted_edges = [edge for edge in weighted_edges if edge[1] not in {"load", "store"}]
    weighted_edges.append(("MEM", "load", 1, 1))
    weighted_edges.append(("MEM", "store", 1, 1))
    return ExecutabilityGraph(
        functional_units=("fu_generic", "MEM"),
        operations=operations,
        edges=tuple(weighted_edges),
    )


def _vendor_memory_components() -> dict[str, dict[str, object]]:
    return validate_component_library(
        {
            "RM_IHPSG13_1P_64x64_c2_bm_bist": {
                "kind": "memory",
                "memory": {"word_t": "i64", "word_len": 64},
                "ports": {
                    "addr": {"dir": "input", "type": "i32"},
                    "wdata": {"dir": "input", "type": "i64"},
                    "we": {"dir": "input", "type": "i1"},
                    "rdata": {"dir": "output", "type": "i64"},
                },
                "supports": {
                    "load": {"ii": 1, "d": 2},
                    "store": {"ii": 1, "d": 1},
                },
            }
        }
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

    def test_lower_seq_to_alloc_infers_semantic_base_type_parameters_per_node(self) -> None:
        seq_design = lower_module_to_seq(
            Module(
                functions=[
                    Function(
                        name="typed_div",
                        params=[
                            Parameter("ai", "i8"),
                            Parameter("bi", "i8"),
                            Parameter("ax", "i32"),
                            Parameter("bx", "i32"),
                        ],
                        return_type="i32",
                        blocks=[
                            Block(
                                "entry",
                                instructions=[
                                    BinaryOp("div", "v1", "i8", "ai", "bi"),
                                    BinaryOp("div", "v2", "i32", "ax", "bx"),
                                ],
                                terminator=ReturnOp("v2"),
                            )
                        ],
                    )
                ],
            )
        )

        operations = tuple(sorted(COMPACT_OPCODE_LABELS))
        graph = ExecutabilityGraph(
            functional_units=("fu_generic", "DIV"),
            operations=operations,
            edges=tuple(("fu_generic", operation, 2, 4) for operation in operations if operation != "div")
            + (("fu_generic", "div", 2, 4), ("DIV", "div", 1, 3)),
            support_types=(("DIV", "div", (("operand0", "base_t"), ("operand1", "base_t"), ("result", "base_t"))),),
        )

        alloc_design = lower_seq_to_alloc(seq_design, executability_graph=graph)
        proc = alloc_design.get_region("proc_typed_div")
        self.assertIsNotNone(proc)
        assert proc is not None

        div_i8 = next(node for node in proc.nodes if node.id == "v1")
        div_i32 = next(node for node in proc.nodes if node.id == "v2")

        self.assertEqual(div_i8.attributes["class"], "DIV<base_t=i8>")
        self.assertEqual(div_i32.attributes["class"], "DIV<base_t=i32>")
        self.assertEqual(div_i8.attributes["delay"], 3)
        self.assertEqual(div_i32.attributes["ii"], 1)

    def test_lower_seq_to_alloc_materializes_resource_type_adapters(self) -> None:
        seq_design = parse_uhir(
            """
            design typed_mul
            stage seq
            input  a : i8
            input  b : i8
            output result : i16

            region proc_typed_mul kind=procedure {
              node v0 = nop role=source
              node v1 = mul a, b : i16
              node v2 = ret v1
              node v3 = nop role=sink
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
            }
            """
        )

        operations = tuple(sorted(COMPACT_OPCODE_LABELS))
        graph = ExecutabilityGraph(
            functional_units=("fu_generic", "MUL32"),
            operations=operations,
            edges=tuple(("fu_generic", operation, 2, 4) for operation in operations if operation != "mul")
            + (("fu_generic", "mul", 2, 4), ("MUL32", "mul", 1, 2)),
            support_types=(("MUL32", "mul", (("operand0", "i32"), ("operand1", "i32"), ("return", "i32"))),),
        )

        alloc_design = lower_seq_to_alloc(seq_design, executability_graph=graph)
        proc = alloc_design.get_region("proc_typed_mul")
        self.assertIsNotNone(proc)
        assert proc is not None

        sext_nodes = [node for node in proc.nodes if node.opcode == "sext"]
        trunc_node = next(node for node in proc.nodes if node.opcode == "trunc")
        mul_node = next(node for node in proc.nodes if node.id == "v1")
        ret_node = next(node for node in proc.nodes if node.id == "v2")

        self.assertEqual(len(sext_nodes), 2)
        self.assertTrue(all(node.attributes["class"] == "ADAPT" for node in [*sext_nodes, trunc_node]))
        self.assertEqual(mul_node.result_type, "i32")
        self.assertEqual(trunc_node.result_type, "i16")
        self.assertEqual(ret_node.operands, (trunc_node.id,))

    def test_lower_seq_to_alloc_uses_region_param_types_for_adapters(self) -> None:
        seq_design = parse_uhir(
            """
            design typed_param
            stage seq
            output result : i16

            region proc_typed_param kind=procedure {
              node v0 = nop role=source params=[a] param_types=[i8]
              node v1 = add a, 0:i16 : i16
              node v2 = ret v1
              node v3 = nop role=sink
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
            }
            """
        )

        operations = tuple(sorted(COMPACT_OPCODE_LABELS))
        graph = ExecutabilityGraph(
            functional_units=("fu_generic", "ALU32"),
            operations=operations,
            edges=tuple(("fu_generic", operation, 2, 4) for operation in operations if operation != "add")
            + (("ALU32", "add", 1, 1),),
            support_types=(("ALU32", "add", (("operand0", "i32"), ("operand1", "i32"), ("return", "i32"))),),
        )

        alloc_design = lower_seq_to_alloc(seq_design, executability_graph=graph)
        proc = alloc_design.get_region("proc_typed_param")
        self.assertIsNotNone(proc)
        assert proc is not None

        param_adapter = next(node for node in proc.nodes if node.opcode == "sext" and node.operands == ("a",))

        self.assertEqual(param_adapter.attributes["source_type"], "i8")
        self.assertEqual(param_adapter.result_type, "i32")

    def test_lower_seq_to_alloc_materializes_memory_load_type_adapters(self) -> None:
        seq_design = parse_uhir(
            """
            design typed_load
            stage seq
            input  A : memref<i32, 4>
            input  i : i8
            output result : i8

            region proc_typed_load kind=procedure {
              node v0 = nop role=source
              node v1 = load A, i : i8
              node v2 = ret v1
              node v3 = nop role=sink
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
            }
            """
        )

        alloc_design = lower_seq_to_alloc(seq_design, executability_graph=_memory_executability_graph())
        proc = alloc_design.get_region("proc_typed_load")
        self.assertIsNotNone(proc)
        assert proc is not None

        addr_adapter = next(node for node in proc.nodes if node.opcode == "sext" and node.operands == ("i",))
        load_node = next(node for node in proc.nodes if node.id == "v1")
        result_adapter = next(node for node in proc.nodes if node.opcode == "trunc" and node.operands == ("v1",))
        ret_node = next(node for node in proc.nodes if node.id == "v2")

        self.assertEqual(addr_adapter.result_type, "i32")
        self.assertEqual(load_node.result_type, "i32")
        self.assertEqual(result_adapter.result_type, "i8")
        self.assertEqual(load_node.operands, ("A", addr_adapter.id))
        self.assertEqual(ret_node.operands, (result_adapter.id,))

    def test_lower_seq_to_alloc_materializes_memory_store_type_adapters(self) -> None:
        seq_design = parse_uhir(
            """
            design typed_store
            stage seq
            input  C : memref<i32, 4>
            input  i : i16
            input  x : i8
            output result : i32

            region proc_typed_store kind=procedure {
              node v0 = nop role=source
              node v1 = store C, i, x
              node v2 = const 0 : i32
              node v3 = ret v2
              node v4 = nop role=sink
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
            }
            """
        )

        alloc_design = lower_seq_to_alloc(seq_design, executability_graph=_memory_executability_graph())
        proc = alloc_design.get_region("proc_typed_store")
        self.assertIsNotNone(proc)
        assert proc is not None

        addr_adapter = next(node for node in proc.nodes if node.opcode == "sext" and node.operands == ("i",))
        data_adapter = next(node for node in proc.nodes if node.opcode == "sext" and node.operands == ("x",))
        store_node = next(node for node in proc.nodes if node.id == "v1")

        self.assertEqual(addr_adapter.attributes["target_type"], "i32")
        self.assertEqual(data_adapter.attributes["source_type"], "i8")
        self.assertEqual(data_adapter.result_type, "i32")
        self.assertEqual(store_node.operands, ("C", addr_adapter.id, data_adapter.id))

    def test_lower_seq_to_alloc_autoram_selects_vendor_memory_macro(self) -> None:
        seq_design = parse_uhir(
            """
            design memdemo
            stage seq
            input  A : memref<i64, 64>
            input  i : i32
            output result : i64

            region proc_memdemo kind=procedure {
              node v0 = nop role=source
              node v1 = load A, i : i64
              node v2 = ret v1
              node v3 = nop role=sink
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
            }
            """
        )

        alloc_design = lower_seq_to_alloc(
            seq_design,
            executability_graph=_memory_executability_graph(),
            memory_policy=MemoryPolicy(mode="autoram", threshold_bits=1024),
            memory_vendor_components=_vendor_memory_components(),
        )
        proc = alloc_design.get_region("proc_memdemo")
        self.assertIsNotNone(proc)
        assert proc is not None
        load_node = next(node for node in proc.nodes if node.opcode == "load")

        self.assertEqual(
            load_node.attributes["class"],
            "RM_IHPSG13_1P_64x64_c2_bm_bist<word_t=i64,word_len=64>",
        )
        self.assertEqual(load_node.attributes["ii"], 1)
        self.assertEqual(load_node.attributes["delay"], 2)

    def test_lower_seq_to_alloc_autoram_falls_back_to_generic_ff_for_unsupported_width(self) -> None:
        seq_design = parse_uhir(
            """
            design memdemo
            stage seq
            input  A : memref<i32, 64>
            input  i : i32
            output result : i32

            region proc_memdemo kind=procedure {
              node v0 = nop role=source
              node v1 = load A, i : i32
              node v2 = ret v1
              node v3 = nop role=sink
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
            }
            """
        )

        alloc_design = lower_seq_to_alloc(
            seq_design,
            executability_graph=_memory_executability_graph(),
            memory_policy=MemoryPolicy(mode="autoram", threshold_bits=1024),
            memory_vendor_components=_vendor_memory_components(),
        )
        proc = alloc_design.get_region("proc_memdemo")
        self.assertIsNotNone(proc)
        assert proc is not None
        load_node = next(node for node in proc.nodes if node.opcode == "load")

        self.assertEqual(load_node.attributes["class"], "GEN_FF_MEM<word_t=i32,word_len=64>")
        self.assertEqual(load_node.attributes["delay"], 1)

    def test_lower_seq_to_alloc_autoram_rejects_unbanked_vendor_memory_overflow(self) -> None:
        seq_design = parse_uhir(
            """
            design memdemo
            stage seq
            input  A : memref<i64, 4096>
            input  i : i32
            output result : i64

            region proc_memdemo kind=procedure {
              node v0 = nop role=source
              node v1 = load A, i : i64
              node v2 = ret v1
              node v3 = nop role=sink
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
            }
            """
        )

        with self.assertRaisesRegex(ValueError, "banked memories are not supported yet"):
            lower_seq_to_alloc(
                seq_design,
                executability_graph=_memory_executability_graph(),
                memory_policy=MemoryPolicy(mode="autoram", threshold_bits=1024),
                memory_vendor_components=_vendor_memory_components(),
            )

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

    def test_lower_seq_to_alloc_accepts_implicit_control_coverage_in_exg(self) -> None:
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

        covered_ops = (
            "add",
            "sub",
            "mul",
            "div",
            "mod",
            "and",
            "or",
            "xor",
            "not",
            "neg",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "shl",
            "shr",
            "mov",
            "load",
            "store",
        )
        graph = ExecutabilityGraph(
            functional_units=("fu_generic",),
            operations=covered_ops,
            edges=tuple(("fu_generic", operation, 1, 1) for operation in covered_ops),
        )

        alloc_design = lower_seq_to_alloc(seq_design, executability_graph=graph)

        proc = alloc_design.get_region("proc_add1")
        assert proc is not None
        ret_node = next(node for node in proc.nodes if node.opcode == "ret")
        self.assertEqual(ret_node.attributes["class"], "CTRL")
        self.assertEqual(ret_node.attributes["delay"], 0)

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
