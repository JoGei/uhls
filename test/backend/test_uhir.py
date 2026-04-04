from __future__ import annotations

import unittest

from uhls.backend.uhir import UHIRParseError, parse_uhir


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

              steps 0..4
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
              port mr0 : memrd A
              port mw0 : memwr C
            }

            region R0 kind=procedure {
              node v1 = load A[i] : i32 class=memrd delay=1 start=0 end=0 bind=mr0
              node v2 = mul v1, 2 : i32 class=mul delay=2 start=1 end=2 bind=mul0
              node v3 = add acc, v2 : i32 class=add delay=1 start=3 end=3 bind=add0
              node v4 = store C[i], v3 class=memwr delay=1 start=4 end=4 bind=mw0

              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4

              value v1 -> r_p live=0..1
              value v3 -> r_acc live=3..4

              mux mx0 : input=[r_acc, r_next] output=r_acc sel=state
            }
            """
        )

        self.assertEqual(design.stage, "bind")
        self.assertEqual([resource.id for resource in design.resources], ["mul0", "add0", "r_acc", "r_p", "mr0", "mw0"])
        region = design.regions[0]
        self.assertEqual(region.nodes[1].attributes["bind"], "mul0")
        self.assertEqual(region.value_bindings[0].producer, "v1")
        self.assertEqual(region.value_bindings[0].register, "r_p")
        self.assertEqual(region.value_bindings[0].live_start, 0)
        self.assertEqual(region.value_bindings[0].live_end, 1)
        self.assertEqual(region.muxes[0].inputs, ("r_acc", "r_next"))
        self.assertEqual(region.muxes[0].output, "r_acc")
        self.assertEqual(region.muxes[0].select, "state")

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
