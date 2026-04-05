from __future__ import annotations

import unittest

from uhls.backend.uhir import create_builtin_gopt_pass, lower_module_to_seq, parse_uhir, project_to_seq_design, run_gopt_passes
from uhls.frontend import lower_source_to_uir
from uhls.middleend.uir import parse_module


class GraphOptimizerTests(unittest.TestCase):
    def test_project_to_seq_design_strips_later_stage_artifacts(self) -> None:
        design = parse_uhir(
            """
            design add1
            stage bind
            schedule kind=control_steps

            resources {
              fu ewms0 : EWMS
              reg r_i32_0 : i32
            }

            region proc_add1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, 1 : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=ewms0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3

              map v1 <- t0_0
              steps [0:1]
              latency 2
              value t0_0 -> r_i32_0 live=[1:1]
            }
            """
        )

        projected = project_to_seq_design(design)
        self.assertEqual(projected.stage, "seq")
        self.assertIsNone(projected.schedule)
        self.assertEqual(projected.resources, [])

        region = projected.get_region("proc_add1")
        self.assertIsNotNone(region)
        assert region is not None
        self.assertIsNone(region.steps)
        self.assertIsNone(region.latency)
        self.assertEqual(region.value_bindings, [])
        node = next(candidate for candidate in region.nodes if candidate.id == "v1")
        self.assertEqual(node.attributes, {})

    def test_infer_static_annotates_frontend_static_loop(self) -> None:
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

        design = run_gopt_passes(
            lower_module_to_seq(module, top="sum4"),
            [create_builtin_gopt_pass("infer_static")],
        )

        proc = design.get_region("proc_sum4")
        self.assertIsNotNone(proc)
        assert proc is not None
        proc_loop = next(node for node in proc.nodes if node.opcode == "loop")
        self.assertEqual(proc_loop.attributes.get("static_trip_count"), 4)

    def test_infer_static_leaves_runtime_bound_loop_unannotated(self) -> None:
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

        design = run_gopt_passes(
            lower_module_to_seq(module, top="sumN"),
            [create_builtin_gopt_pass("infer_static")],
        )

        proc = design.get_region("proc_sumN")
        self.assertIsNotNone(proc)
        assert proc is not None
        proc_loop = next(node for node in proc.nodes if node.opcode == "loop")
        self.assertNotIn("static_trip_count", proc_loop.attributes)

    def test_infer_static_annotates_parsed_uir_loop(self) -> None:
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

        design = run_gopt_passes(
            lower_module_to_seq(module, top="dot4"),
            [create_builtin_gopt_pass("infer_static")],
        )

        proc = design.get_region("proc_dot4")
        self.assertIsNotNone(proc)
        assert proc is not None
        proc_loop = next(node for node in proc.nodes if node.opcode == "loop")
        self.assertEqual(proc_loop.attributes.get("static_trip_count"), 4)

    def test_simplify_static_control_removes_static_loop_compare_artifact(self) -> None:
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

        design = run_gopt_passes(
            lower_module_to_seq(module, top="dot4"),
            [
                create_builtin_gopt_pass("infer_static"),
                create_builtin_gopt_pass("simplify_static_control"),
            ],
        )

        loop_region = next((region for region in design.regions if region.kind == "loop"), None)
        self.assertIsNotNone(loop_region)
        assert loop_region is not None

        self.assertNotIn("lt", [node.opcode for node in loop_region.nodes])
        branch = next(node for node in loop_region.nodes if node.opcode == "branch")
        phi = next(node for node in loop_region.nodes if node.opcode == "phi")
        self.assertEqual(branch.operands, ())
        self.assertNotIn("t0_0", [mapping.source_id for mapping in loop_region.mappings])
        loop_data_edges = {(edge.source, edge.target) for edge in loop_region.edges if edge.kind == "data"}
        self.assertIn((phi.id, branch.id), loop_data_edges)
