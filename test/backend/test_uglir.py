from __future__ import annotations

import json
import unittest
from pathlib import Path

from uhls.backend.hls import lower_alloc_to_sched, lower_bind_to_fsm, lower_fsm_to_uglir, lower_sched_to_bind
from uhls.backend.hls.bind.builtin.left_edge import LeftEdgeBinder
from uhls.backend.hls.uglir import format_uglir, parse_uglir
from uhls.backend.hls.uglir.lower import _producer_global_capture_steps, _producer_region, _value_global_live_starts
from uhls.backend.hls.uhir import (
    ExecutabilityGraph,
    create_builtin_gopt_pass,
    lower_module_to_seq,
    lower_seq_to_alloc,
    parse_uhir,
    run_gopt_passes,
)
from uhls.frontend import lower_source_to_uir
from uhls.middleend.passes.opt import (
    CSEPass,
    CanonicalizeLoopsPass,
    ConstPropPass,
    CopyPropPass,
    DCEPass,
    InlineCallsPass,
    MovToAddZeroPass,
    PruneFunctionsPass,
    UnrollLoopsPass,
)
from uhls.middleend.passes.util import PassContext, PassManager


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
        "sel",
    )
    return ExecutabilityGraph(
        functional_units=("EWMS",),
        operations=operations,
        edges=tuple(("EWMS", operation, 1, 1) for operation in operations),
    )


def _generic_component_library() -> dict[str, dict[str, object]]:
    return json.loads(Path("src/uhls/backend/hls/impl/generic/gen.uhlslib.json").read_text(encoding="utf-8"))["components"]


def _executability_graph_from_component_library(
    component_library: dict[str, dict[str, object]]
) -> ExecutabilityGraph:
    functional_units: list[str] = []
    operations: dict[str, None] = {}
    edges: list[tuple[str, str, int, int]] = []
    for component_name, component in component_library.items():
        supports = component.get("supports")
        if not isinstance(supports, dict):
            continue
        functional_units.append(component_name)
        for operation_name, support in supports.items():
            if not isinstance(support, dict):
                continue
            ii = support.get("ii")
            delay = support.get("d")
            if not isinstance(ii, int) or not isinstance(delay, int):
                continue
            operations.setdefault(str(operation_name), None)
            edges.append((component_name, str(operation_name), ii, delay))
    return ExecutabilityGraph(
        functional_units=tuple(functional_units),
        operations=tuple(operations),
        edges=tuple(edges),
    )


def _lower_unrolled_example_to_uglir(
    stem: str,
    *,
    factor: int,
    optimize: bool = True,
    cleanup_after_unroll: bool = True,
    component_library: dict[str, dict[str, object]] | None = None,
    flatten: bool = False,
):
    source = Path(f"examples/{stem}/{stem}.c").read_text(encoding="utf-8")
    optimized_module = PassManager(
        [
            InlineCallsPass(),
            PruneFunctionsPass(),
            ConstPropPass(),
        ]
    ).run(lower_source_to_uir(source), PassContext(pass_args=(stem,)))
    if optimize:
        optimized_module = PassManager(
            [
                DCEPass(),
                CSEPass(),
                CopyPropPass(),
                DCEPass(),
            ]
        ).run(optimized_module)
    post_unroll_pipeline = [UnrollLoopsPass("1", factor), CanonicalizeLoopsPass()]
    if cleanup_after_unroll:
        post_unroll_pipeline.extend([ConstPropPass(), CopyPropPass(), DCEPass()])
    optimized_module = PassManager(post_unroll_pipeline).run(optimized_module)
    seq_design = run_gopt_passes(
        lower_module_to_seq(optimized_module, top=stem),
        [
            create_builtin_gopt_pass("infer_loops"),
            create_builtin_gopt_pass("translate_loop_dialect"),
            create_builtin_gopt_pass("infer_static"),
            create_builtin_gopt_pass("simplify_static_control"),
            create_builtin_gopt_pass("predicate"),
            create_builtin_gopt_pass("fold_predicates"),
        ],
    )
    executability_graph = (
        _full_executability_graph()
        if component_library is None
        else _executability_graph_from_component_library(component_library)
    )
    alloc_design = lower_seq_to_alloc(seq_design, executability_graph=executability_graph)
    sched_design = lower_alloc_to_sched(alloc_design)
    bind_design = lower_sched_to_bind(sched_design, binder=LeftEdgeBinder(flatten=flatten))
    return lower_fsm_to_uglir(lower_bind_to_fsm(bind_design), component_library=component_library)


def _lower_unrolled_dot4_relu_to_uglir():
    return _lower_unrolled_example_to_uglir("dot4_relu", factor=2)


def _lower_example_to_uglir(
    stem: str,
    *,
    optimize: bool = True,
    legalize_mov_to_add_zero: bool = False,
):
    source = Path(f"examples/{stem}/{stem}.c").read_text(encoding="utf-8")
    legalize_pipeline = [
        InlineCallsPass(),
        PruneFunctionsPass(),
        ConstPropPass(),
    ]
    if legalize_mov_to_add_zero:
        legalize_pipeline.append(MovToAddZeroPass())
    optimized_module = PassManager(legalize_pipeline).run(lower_source_to_uir(source), PassContext(pass_args=(stem,)))
    if optimize:
        optimized_module = PassManager(
            [
                DCEPass(),
                CSEPass(),
                CopyPropPass(),
                DCEPass(),
            ]
        ).run(optimized_module)
    seq_design = run_gopt_passes(
        lower_module_to_seq(optimized_module, top=stem),
        [
            create_builtin_gopt_pass("infer_loops"),
            create_builtin_gopt_pass("translate_loop_dialect"),
            create_builtin_gopt_pass("infer_static"),
            create_builtin_gopt_pass("simplify_static_control"),
            create_builtin_gopt_pass("predicate"),
            create_builtin_gopt_pass("fold_predicates"),
        ],
    )
    alloc_design = lower_seq_to_alloc(seq_design, executability_graph=_full_executability_graph())
    sched_design = lower_alloc_to_sched(alloc_design)
    bind_design = lower_sched_to_bind(sched_design)
    return lower_fsm_to_uglir(lower_bind_to_fsm(bind_design))


def _lower_example_to_fsm(
    stem: str,
    *,
    optimize: bool = True,
    legalize_mov_to_add_zero: bool = False,
):
    source = Path(f"examples/{stem}/{stem}.c").read_text(encoding="utf-8")
    legalize_pipeline = [
        InlineCallsPass(),
        PruneFunctionsPass(),
        ConstPropPass(),
    ]
    if legalize_mov_to_add_zero:
        legalize_pipeline.append(MovToAddZeroPass())
    optimized_module = PassManager(legalize_pipeline).run(lower_source_to_uir(source), PassContext(pass_args=(stem,)))
    if optimize:
        optimized_module = PassManager(
            [
                DCEPass(),
                CSEPass(),
                CopyPropPass(),
                DCEPass(),
            ]
        ).run(optimized_module)
    seq_design = run_gopt_passes(
        lower_module_to_seq(optimized_module, top=stem),
        [
            create_builtin_gopt_pass("infer_loops"),
            create_builtin_gopt_pass("translate_loop_dialect"),
            create_builtin_gopt_pass("infer_static"),
            create_builtin_gopt_pass("simplify_static_control"),
            create_builtin_gopt_pass("predicate"),
            create_builtin_gopt_pass("fold_predicates"),
        ],
    )
    alloc_design = lower_seq_to_alloc(seq_design, executability_graph=_full_executability_graph())
    sched_design = lower_alloc_to_sched(alloc_design)
    bind_design = lower_sched_to_bind(sched_design)
    return lower_bind_to_fsm(bind_design)


class UGLIRLoweringTests(unittest.TestCase):
    """Coverage for fsm-to-uglir lowering."""

    def test_lower_fsm_to_uglir_emits_static_glue_shell(self) -> None:
        fsm_design = parse_uhir(
            """
            design add1
            stage fsm
            schedule kind=control_steps
            input  x : i32
            output result : i32
            resources {
              fu ewms0 : EWMS
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_add1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state DONE code=16
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[ewms0<-v1]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_add1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, 1:i32 : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=ewms0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
              value v1 -> r_i32_0 live=[1:1]
            }
            """
        )

        uglir_design = lower_fsm_to_uglir(fsm_design)

        self.assertEqual(uglir_design.stage, "uglir")
        self.assertEqual([(port.name, port.type) for port in uglir_design.inputs[:4]], [("clk", "clock"), ("rst", "i1"), ("req_valid", "i1"), ("resp_ready", "i1")])
        self.assertEqual([(port.name, port.type) for port in uglir_design.outputs[:3]], [("req_ready", "i1"), ("resp_valid", "i1"), ("result", "i32")])
        resource_ids = [resource.id for resource in uglir_design.resources]
        self.assertIn("state_q", resource_ids)
        self.assertIn("next_state_n", resource_ids)
        self.assertIn("ewms0", resource_ids)
        self.assertIn("ewms0_go_n", resource_ids)
        self.assertIn("ewms0_y_n", resource_ids)
        self.assertIn("mx_r_i32_0_n", resource_ids)
        assign_pairs = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("req_fire_n", "req_valid & req_ready"), assign_pairs)
        self.assertIn(("resp_fire_n", "resp_valid & resp_ready"), assign_pairs)
        self.assertIn(("req_ready", "state_q == 1"), assign_pairs)
        self.assertIn(("resp_valid", "state_q == 16"), assign_pairs)
        self.assertIn(
            ("next_state_n", "(state_q == 1 && req_fire_n) ? 2 : state_q == 2 ? 4 : state_q == 4 ? 8 : state_q == 8 ? 16 : (state_q == 16 && resp_fire_n) ? 1 : 1"),
            assign_pairs,
        )
        self.assertIn(("ewms0", "go", "ewms0_go_n"), [(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments])
        self.assertIn(("ewms0", "y", "ewms0_y_n"), [(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments])
        mux = uglir_design.muxes[0]
        self.assertEqual(mux.name, "mx_r_i32_0_n")
        self.assertEqual(mux.select, "sel_r_i32_0_n")
        self.assertEqual([(case.key, case.source) for case in mux.cases], [("HOLD", "r_i32_0_q"), ("SRC_V1", "v1_n")])
        seq_block = uglir_design.seq_blocks[0]
        self.assertEqual(seq_block.clock, "clk")
        self.assertEqual(seq_block.reset, "rst")
        self.assertEqual([(update.target, update.value, update.enable) for update in seq_block.reset_updates], [("state_q", "1", None)])
        self.assertEqual([(update.target, update.value, update.enable) for update in seq_block.updates], [("state_q", "next_state_n", None), ("r_i32_0_q", "mx_r_i32_0_n", "latch_r_i32_0_n")])
        rendered = format_uglir(uglir_design)
        self.assertIn("stage uglir", rendered)
        self.assertIn("input  req_valid : i1", rendered)
        self.assertIn("output req_ready : i1", rendered)
        self.assertIn("inst ewms0 : EWMS", rendered)
        self.assertIn("assign req_ready = state_q == 1", rendered)
        self.assertIn("ewms0.go(ewms0_go_n)", rendered)
        self.assertIn("mux mx_r_i32_0_n : i32 sel=sel_r_i32_0_n {", rendered)
        self.assertIn("seq clk {", rendered)

    def test_parse_and_format_uglir_preserve_component_library_provenance(self) -> None:
        design = parse_uglir(
            """
            design add1
            stage uglir
            component_library "../lib/gen.uhlslib.json"
            component_library "/abs/vendor.uhlslib.json"
            input  clk : clock
            output y : i32
            resources {
              net y_n : i32
            }
            assign y = 0:i32
            """
        )

        self.assertEqual(
            design.component_libraries,
            ["../lib/gen.uhlslib.json", "/abs/vendor.uhlslib.json"],
        )
        rendered = format_uglir(design)
        self.assertIn('component_library "../lib/gen.uhlslib.json"', rendered)
        self.assertIn('component_library "/abs/vendor.uhlslib.json"', rendered)
        reparsed = parse_uglir(rendered)
        self.assertEqual(reparsed.component_libraries, design.component_libraries)

    def test_lower_fsm_to_uglir_uses_component_library_ports(self) -> None:
        fsm_design = parse_uhir(
            """
            design add1
            stage fsm
            schedule kind=control_steps
            input  x : i32
            output result : i32
            resources {
              fu ewms0 : ALU
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_add1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state DONE code=16
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[ewms0<-v1]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_add1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, 1:i32 : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=ewms0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
              value v1 -> r_i32_0 live=[1:1]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "ALU": {
                  "kind": "combinational",
                  "ports": {
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "op": { "dir": "input", "type": "u5" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "add": {
                      "ii": 1,
                      "d": 1,
                      "opcode": 0,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        resource_ids = [resource.id for resource in uglir_design.resources]
        self.assertIn("ewms0_a_q", resource_ids)
        self.assertIn("ewms0_b_q", resource_ids)
        self.assertIn("ewms0_op_q", resource_ids)
        self.assertIn("ewms0_y_n", resource_ids)
        self.assertIn("sel_ewms0_a_n", resource_ids)
        self.assertIn("sel_ewms0_b_n", resource_ids)
        self.assertIn("mx_ewms0_a_n", resource_ids)
        self.assertIn("mx_ewms0_b_n", resource_ids)
        self.assertNotIn("ewms0_go_n", resource_ids)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertIn(("ewms0", "a", "ewms0_a_q"), attachments)
        self.assertIn(("ewms0", "b", "ewms0_b_q"), attachments)
        self.assertIn(("ewms0", "op", "ewms0_op_q"), attachments)
        self.assertIn(("ewms0", "y", "ewms0_y_n"), attachments)
        assign_pairs = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("sel_ewms0_a_n", "state_q == 2 ? SRC_X : ZERO"), assign_pairs)
        self.assertIn(("sel_ewms0_b_n", "state_q == 2 ? CONST_1_I32 : ZERO"), assign_pairs)
        self.assertIn(("sel_ewms0_op_n", "state_q == 2 ? SRC_0 : ZERO"), assign_pairs)
        mux_a = next(mux for mux in uglir_design.muxes if mux.name == "mx_ewms0_a_n")
        mux_b = next(mux for mux in uglir_design.muxes if mux.name == "mx_ewms0_b_n")
        self.assertEqual([(case.key, case.source) for case in mux_a.cases], [("ZERO", "const_i32_0_n"), ("SRC_X", "x")])
        self.assertEqual([(case.key, case.source) for case in mux_b.cases], [("ZERO", "const_i32_0_n"), ("CONST_1_I32", "const_i32_1_n")])
        rendered = format_uglir(uglir_design)
        self.assertIn("ewms0.a(ewms0_a_q)", rendered)
        self.assertIn("ewms0.b(ewms0_b_q)", rendered)
        self.assertIn("mux mx_ewms0_a_n : i32 sel=sel_ewms0_a_n {", rendered)
        self.assertIn("ewms0.op(ewms0_op_q)", rendered)
        self.assertNotIn("ewms0.go(", rendered)

    def test_lower_fsm_to_uglir_resolves_static_child_call_operands_from_callsite(self) -> None:
        fsm_design = parse_uhir(
            """
            design child_call
            stage fsm
            schedule kind=hierarchical
            input  x : i32
            output result : i32
            resources {
              fu alu0 : ALU
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_top {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state DONE code=8
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[alu0<-c1]
              emit DONE resp_valid=true
            }

            region proc_top kind=procedure {
              region_ref proc_child
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = call child, x : i32 child=proc_child class=CTRL ii=1 delay=1 start=0 end=0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
            }

            region proc_child kind=procedure {
              node c0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node c1 = add c_0, 1:i32 : i32 class=ALU ii=1 delay=1 start=0 end=0 bind=alu0
              node c2 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
              edge data c0 -> c1
              edge data c1 -> c2
              steps [0:0]
              latency 1
              value c1 -> r_i32_0 live=[1:1]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "ALU": {
                  "kind": "combinational",
                  "ports": {
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "op": { "dir": "input", "type": "u5" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "add": {
                      "ii": 1,
                      "d": 1,
                      "opcode": 0,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        rendered = format_uglir(uglir_design)
        self.assertNotIn("c_0", rendered)
        self.assertIn("SRC_X", rendered)

    def test_lower_fsm_to_uglir_keeps_duplicated_source_ids_region_local(self) -> None:
        fsm_design = parse_uhir(
            """
            design child_call_shadow
            stage fsm
            schedule kind=hierarchical
            input  x : i32
            output result : i32
            resources {
              fu alu0 : ALU
              fu alu1 : ALU
              reg r_i32_0 : i32
              reg r_i32_1 : i32
            }
            controller C0 encoding=binary protocol=req_resp completion_order=in_order overlap=true region=proc_top {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=0
              state T0 code=1
              state T1 code=2
              state T2 code=3
              state DONE code=4
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[alu0<-v1]
              emit T1 issue=[alu1<-c1] latch=[r_i32_0]
              emit T2 latch=[r_i32_1]
              emit DONE resp_valid=true
            }

            region proc_top kind=procedure {
              region_ref proc_child
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, 2:i32 : i32 class=ALU ii=1 delay=1 start=0 end=0 bind=alu0
              node v2 = call child, t1_0 : i32 child=proc_child class=CTRL ii=1 delay=1 start=1 end=1
              node v3 = ret v2 class=CTRL ii=0 delay=0 start=2 end=2
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
              map v1 <- t1_0
              steps [0:2]
              latency 3
              value t1_0 -> r_i32_0 live=[1:1]
            }

            region proc_child kind=procedure {
              node c0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
              node c1 = add c_0, 1:i32 : i32 class=ALU ii=1 delay=1 start=1 end=1 bind=alu1
              node c2 = ret t1_0 class=CTRL ii=0 delay=0 start=2 end=2
              node c3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data c0 -> c1
              edge data c1 -> c2
              edge data c2 -> c3
              map c1 <- t1_0
              steps [1:2]
              latency 2
              value t1_0 -> r_i32_1 live=[2:2]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "ALU": {
                  "kind": "combinational",
                  "ports": {
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "op": { "dir": "input", "type": "u5" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "add": {
                      "ii": 1,
                      "d": 1,
                      "opcode": 0,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        rendered = format_uglir(uglir_design)
        self.assertIn("SRC_ALU0_Y", rendered)
        self.assertIn("alu0_y", rendered)
        self.assertNotIn("alu1_a(t1_0", rendered)

    def test_lower_fsm_to_uglir_uses_component_issue_and_result_bindings(self) -> None:
        fsm_design = parse_uhir(
            """
            design seqadd1
            stage fsm
            schedule kind=control_steps
            input  x : i32
            output result : i32
            resources {
              fu seq0 : SEQADD
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_seqadd1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state DONE code=16
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[seq0<-v1]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_seqadd1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = add x, 1:i32 : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=seq0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
              value v1 -> r_i32_0 live=[1:1]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "SEQADD": {
                  "kind": "sequential",
                  "issue": {
                    "start": "true"
                  },
                  "ports": {
                    "start": { "dir": "input", "type": "i1" },
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "out": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "add": {
                      "ii": 1,
                      "d": 1,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "out": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        resource_ids = [resource.id for resource in uglir_design.resources]
        self.assertIn("seq0_start_n", resource_ids)
        self.assertIn("seq0_a_n", resource_ids)
        self.assertIn("seq0_b_n", resource_ids)
        self.assertIn("seq0_out_n", resource_ids)
        self.assertNotIn("seq0_go_n", resource_ids)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertIn(("seq0", "start", "seq0_start_n"), attachments)
        self.assertIn(("seq0", "out", "seq0_out_n"), attachments)
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("seq0_start_n", "state_q == 2"), assigns)
        self.assertIn(("sel_seq0_a_n", "state_q == 2 ? SRC_X : ZERO"), assigns)
        self.assertIn(("sel_seq0_b_n", "state_q == 2 ? CONST_1_I32 : ZERO"), assigns)
        mux_a = next(mux for mux in uglir_design.muxes if mux.name == "mx_seq0_a_n")
        mux_b = next(mux for mux in uglir_design.muxes if mux.name == "mx_seq0_b_n")
        self.assertEqual([(case.key, case.source) for case in mux_a.cases], [("ZERO", "const_i32_0_n"), ("SRC_X", "x")])
        self.assertEqual([(case.key, case.source) for case in mux_b.cases], [("ZERO", "const_i32_0_n"), ("CONST_1_I32", "const_i32_1_n")])
        mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_r_i32_0_n")
        self.assertIn(("SRC_V1", "v1_n"), [(case.key, case.source) for case in mux.cases])
        rendered = format_uglir(uglir_design)
        self.assertIn("assign seq0_start_n = state_q == 2", rendered)
        self.assertIn("assign seq0_a_n = mx_seq0_a_n", rendered)
        self.assertIn("assign seq0_b_n = mx_seq0_b_n", rendered)
        self.assertIn("seq0.a(seq0_a_n)", rendered)
        self.assertIn("seq0.b(seq0_b_n)", rendered)
        self.assertIn("seq0.start(seq0_start_n)", rendered)
        self.assertIn("seq0.out(seq0_out_n)", rendered)
        self.assertNotIn("seq0.go(", rendered)

    def test_lower_fsm_to_uglir_captures_pipelined_result_before_first_consumer_cycle(self) -> None:
        fsm_design = parse_uhir(
            """
            design pipe_use
            stage fsm
            schedule kind=control_steps
            input  x : i32
            input  y : i32
            output result : i32
            resources {
              fu mul0 : PMUL
              fu alu0 : ALU
              reg r_i32_0 : i32
              reg r_i32_1 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_pipe_use {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state T3 code=16
              state DONE code=32
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> T3
              transition T3 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mul0<-v1]
              emit T1 latch=[r_i32_0]
              emit T2 issue=[alu0<-v2]
              emit T3 latch=[r_i32_1]
              emit DONE resp_valid=true
            }

            region proc_pipe_use kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = mul x, y : i32 class=PMUL ii=1 delay=2 start=0 end=1 bind=mul0
              node v2 = add v1, 1:i32 : i32 class=ALU ii=1 delay=1 start=2 end=2 bind=alu0
              node v3 = ret v2 class=CTRL ii=0 delay=0 start=3 end=3
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
              steps [0:3]
              latency 4
              value v1 -> r_i32_0 live=[2:2]
              value v2 -> r_i32_1 live=[3:3]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "PMUL": {
                  "kind": "pipelined",
                  "ports": {
                    "clk": { "dir": "input", "type": "clock" },
                    "rst": { "dir": "input", "type": "reset", "kind": "sync", "active": "hi" },
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "mul": {
                      "ii": 1,
                      "d": 2,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                },
                "ALU": {
                  "kind": "combinational",
                  "ports": {
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "op": { "dir": "input", "type": "u5" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "add": {
                      "ii": 1,
                      "d": 1,
                      "opcode": 0,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        assign_pairs = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("latch_r_i32_0_n", "state_q == 4"), assign_pairs)
        self.assertIn(("sel_r_i32_0_n", "state_q == 4 ? SRC_V1 : HOLD"), assign_pairs)
        self.assertIn(("sel_alu0_a_n", "state_q == 8 ? SRC_R_I32_0 : ZERO"), assign_pairs)

        mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_alu0_a_n")
        self.assertEqual(
            [(case.key, case.source) for case in mux.cases],
            [("ZERO", "const_i32_0_n"), ("SRC_R_I32_0", "r_i32_0_q")],
        )

        rendered = format_uglir(uglir_design)
        self.assertIn("assign latch_r_i32_0_n = state_q == 4", rendered)
        self.assertIn("assign sel_r_i32_0_n = state_q == 4 ? SRC_V1 : HOLD", rendered)
        self.assertIn("assign sel_alu0_a_n = state_q == 8 ? SRC_R_I32_0 : ZERO", rendered)

    def test_lower_fsm_to_uglir_lowers_recursive_controller_links_to_nets(self) -> None:
        fsm_design = parse_uhir(
            """
            design dyn_call
            stage fsm
            schedule kind=hierarchical
            resources {
              fu fu_add0 : FU_ADD
            }
            controller C0 encoding=binary protocol=req_resp completion_order=in_order overlap=true region=proc_dyn_call {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=0
              state P0 code=1
              state WAIT_v1 code=2
              state DONE code=3
              transition IDLE -> P0 when=req_valid && req_ready && symb_ready_v1
              transition P0 -> WAIT_v1
              transition WAIT_v1 -> DONE when=symb_done_v1
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit P0 activate=[v1]
              emit DONE resp_valid=true
              link C_callee via=v1 act=[activate, act_valid] ready=[symb_ready_v1, act_ready] done=[symb_done_v1, done_valid] done_ready=[resp_ready, done_ready]
            }

            controller C_callee encoding=binary protocol=act_done completion_order=in_order overlap=true region=callee parent_node=v1 {
              input  act_valid : i1
              input  done_ready : i1
              output act_ready : i1
              output done_valid : i1
              state IDLE code=0
              state T0 code=1
              state DONE code=2
              transition IDLE -> T0 when=act_valid && act_ready
              transition T0 -> DONE
              transition DONE -> IDLE when=done_valid && done_ready
              emit IDLE act_ready=true
              emit DONE done_valid=true
            }

            region proc_dyn_call kind=procedure {
              region_ref callee
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = call child=callee class=CTRL ii=1 delay=1 start=0 end=0 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done
              node v2 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
              edge data v0 -> v1
              edge data v1 -> v2
              edge seq v1 -> callee hierarchy=true
              edge seq callee -> v1 hierarchy=true
            }

            region callee kind=procedure parent=proc_dyn_call {
              node c0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node c1 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
              edge data c0 -> c1
            }
            """
        )

        uglir_design = lower_fsm_to_uglir(fsm_design)

        resource_ids = [resource.id for resource in uglir_design.resources]
        self.assertIn("state_q", resource_ids)
        self.assertIn("next_state_n", resource_ids)
        self.assertIn("C_callee_state_q", resource_ids)
        self.assertIn("C_callee_next_state_n", resource_ids)
        self.assertIn("C_callee_act_valid_n", resource_ids)
        self.assertIn("C_callee_done_ready_n", resource_ids)
        self.assertIn("C_callee_act_ready_n", resource_ids)
        self.assertIn("C_callee_done_valid_n", resource_ids)
        self.assertIn("symb_ready_v1_n", resource_ids)
        self.assertIn("symb_done_v1_n", resource_ids)
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("C_callee_act_valid_n", "state_q == 1"), assigns)
        self.assertIn(("C_callee_done_ready_n", "resp_ready"), assigns)
        self.assertIn(("symb_ready_v1_n", "C_callee_act_ready_n"), assigns)
        self.assertIn(("symb_done_v1_n", "C_callee_done_valid_n"), assigns)
        self.assertIn(("C_callee_act_ready_n", "C_callee_state_q == 0"), assigns)
        self.assertIn(("C_callee_done_valid_n", "C_callee_state_q == 2"), assigns)
        rendered = format_uglir(uglir_design)
        self.assertIn("reg C_callee_state_q : u2", rendered)
        self.assertIn("net C_callee_act_valid_n : i1", rendered)
        self.assertIn("assign symb_done_v1_n = C_callee_done_valid_n", rendered)

    def test_lower_fsm_to_uglir_expands_memref_interface_for_memory_component(self) -> None:
        fsm_design = parse_uhir(
            """
            design read1
            stage fsm
            schedule kind=control_steps
            input  A : memref<i32, 4>
            input  i : i32
            output result : i32
            resources {
              fu mr0 : MEM
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_read1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state DONE code=16
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mr0<-v1]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_read1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = load A, i : i32 class=MEM ii=1 delay=1 start=0 end=0 bind=mr0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
              value v1 -> r_i32_0 live=[1:1]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "MEM": {
                  "kind": "memory",
                  "ports": {
                    "addr": { "dir": "input", "type": "i32" },
                    "wdata": { "dir": "input", "type": "i32" },
                    "we": { "dir": "input", "type": "i1" },
                    "rdata": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "load": {
                      "ii": 1,
                      "d": 1,
                      "bind": {
                        "addr": "operand1",
                        "rdata": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        self.assertEqual([(port.name, port.type) for port in uglir_design.inputs[:6]], [("clk", "clock"), ("rst", "i1"), ("req_valid", "i1"), ("resp_ready", "i1"), ("A_rdata", "i32"), ("i", "i32")])
        self.assertEqual([(port.name, port.type) for port in uglir_design.outputs[:4]], [("req_ready", "i1"), ("resp_valid", "i1"), ("result", "i32"), ("A_addr", "i32")])
        self.assertIn(
            ("port", "A", "MEM<word_t=i32,word_len=4>", "A"),
            [(resource.kind, resource.id, resource.value, resource.target) for resource in uglir_design.resources],
        )
        self.assertNotIn(("inst", "mr0", "MEM", None), [(resource.kind, resource.id, resource.value, resource.target) for resource in uglir_design.resources])
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("sel_mr0_addr_n", "state_q == 2 ? SRC_I : ZERO"), assigns)
        self.assertIn(("A_addr", "mr0_addr_q"), assigns)
        self.assertIn(("mr0_rdata_n", "A_rdata"), assigns)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertNotIn(("mr0", "addr", "mr0_addr_n"), attachments)
        self.assertNotIn(("mr0", "rdata", "mr0_rdata_n"), attachments)
        addr_mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_mr0_addr_n")
        self.assertEqual([(case.key, case.source) for case in addr_mux.cases], [("ZERO", "const_i32_0_n"), ("SRC_I", "i")])
        rendered = format_uglir(uglir_design)
        self.assertIn("input  A_rdata : i32", rendered)
        self.assertIn("output A_addr : i32", rendered)
        self.assertIn("port A : MEM<word_t=i32,word_len=4> A", rendered)
        self.assertIn("assign mr0_rdata_n = A_rdata", rendered)
        self.assertIn("assign A_addr = mr0_addr_q", rendered)

    def test_lower_fsm_to_uglir_expands_store_memref_outputs_for_memory_component(self) -> None:
        fsm_design = parse_uhir(
            """
            design write1
            stage fsm
            schedule kind=control_steps
            input  C : memref<i32>
            input  i : i32
            input  x : i32
            resources {
              fu mw0 : MEM
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_write1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state DONE code=8
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mw0<-v1]
              emit DONE resp_valid=true
            }

            region proc_write1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = store C, i, x class=MEM ii=1 delay=1 start=0 end=0 bind=mw0
              node v2 = ret 0:i32 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "MEM": {
                  "kind": "memory",
                  "ports": {
                    "addr": { "dir": "input", "type": "i32" },
                    "wdata": { "dir": "input", "type": "i32" },
                    "we": { "dir": "input", "type": "i1" },
                    "rdata": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "store": {
                      "ii": 1,
                      "d": 1,
                      "mode": "write",
                      "bind": {
                        "addr": "operand1",
                        "wdata": "operand2",
                        "we": "true"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        self.assertEqual(
            [(port.name, port.type) for port in uglir_design.outputs[:5]],
            [("req_ready", "i1"), ("resp_valid", "i1"), ("C_addr", "i32"), ("C_wdata", "i32"), ("C_we", "i1")],
        )
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("sel_mw0_addr_n", "state_q == 2 ? SRC_I : ZERO"), assigns)
        self.assertIn(("sel_mw0_wdata_n", "state_q == 2 ? SRC_X : ZERO"), assigns)
        self.assertIn(("sel_mw0_we_n", "state_q == 2 ? TRUE : FALSE"), assigns)
        self.assertIn(("C_addr", "mw0_addr_q"), assigns)
        self.assertIn(("C_wdata", "mw0_wdata_q"), assigns)
        self.assertIn(("C_we", "mw0_we_q"), assigns)
        attachments = {(attachment.instance, attachment.port, attachment.signal) for attachment in uglir_design.attachments}
        self.assertNotIn(("mw0", "addr", "mw0_addr_n"), attachments)
        self.assertNotIn(("mw0", "wdata", "mw0_wdata_n"), attachments)
        self.assertNotIn(("mw0", "we", "mw0_we_n"), attachments)
        addr_mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_mw0_addr_n")
        wdata_mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_mw0_wdata_n")
        we_mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_mw0_we_n")
        self.assertEqual([(case.key, case.source) for case in addr_mux.cases], [("ZERO", "const_i32_0_n"), ("SRC_I", "i")])
        self.assertEqual([(case.key, case.source) for case in wdata_mux.cases], [("ZERO", "const_i32_0_n"), ("SRC_X", "x")])
        self.assertEqual([(case.key, case.source) for case in we_mux.cases], [("FALSE", "const_i1_0_n"), ("TRUE", "const_i1_1_n")])
        rendered = format_uglir(uglir_design)
        self.assertIn("output C_addr : i32", rendered)
        self.assertIn("output C_wdata : i32", rendered)
        self.assertIn("output C_we : i1", rendered)
        self.assertIn("assign C_addr = mw0_addr_q", rendered)
        self.assertIn("assign C_wdata = mw0_wdata_q", rendered)
        self.assertIn("assign C_we = mw0_we_q", rendered)

    def test_lower_fsm_to_uglir_honors_component_tied_input_ports(self) -> None:
        fsm_design = parse_uhir(
            """
            design read1
            stage fsm
            schedule kind=control_steps
            input  A : memref<i32, 4>
            input  i : i32
            output result : i32
            resources {
              fu mr0 : MEM
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_read1 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state DONE code=16
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mr0<-v1]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_read1 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = load A, i : i32 class=MEM ii=1 delay=1 start=0 end=0 bind=mr0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:1]
              latency 2
              value v1 -> r_i32_0 live=[1:1]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "MEM": {
                  "kind": "memory",
                  "ports": {
                    "addr": { "dir": "input", "type": "i32" },
                    "bist_en": { "dir": "input", "type": "i1", "tie": "false" },
                    "wdata": { "dir": "input", "type": "i32" },
                    "we": { "dir": "input", "type": "i1" },
                    "rdata": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "load": {
                      "ii": 1,
                      "d": 1,
                      "bind": {
                        "addr": "operand1",
                        "rdata": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("mr0_bist_en_n", "false"), assigns)
        self.assertNotIn(("sel_mr0_bist_en_n", "FALSE"), assigns)
        self.assertNotIn("mx_mr0_bist_en_n", [mux.name for mux in uglir_design.muxes])

    def test_lower_fsm_to_uglir_time_multiplexes_multiple_loads_onto_one_memory_interface(self) -> None:
        fsm_design = parse_uhir(
            """
            design read2
            stage fsm
            schedule kind=control_steps
            input  A : memref<i32>
            input  i : i32
            input  j : i32
            output result : i32
            resources {
              fu mr0 : MEM
              reg r_i32_0 : i32
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_read2 {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state T1 code=4
              state T2 code=8
              state T3 code=16
              state DONE code=32
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> T3
              transition T3 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mr0<-v1]
              emit T1 latch=[r_i32_0]
              emit T2 issue=[mr0<-v2]
              emit T3 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_read2 kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = load A, i : i32 class=MEM ii=1 delay=1 start=0 end=0 bind=mr0
              node v2 = load A, j : i32 class=MEM ii=1 delay=1 start=2 end=2 bind=mr0
              node v3 = ret v2 class=CTRL ii=0 delay=0 start=3 end=3
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
              steps [0:3]
              latency 4
              value v1 -> r_i32_0 live=[1:2]
              value v2 -> r_i32_0 live=[3:3]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "MEM": {
                  "kind": "memory",
                  "ports": {
                    "addr": { "dir": "input", "type": "i32" },
                    "wdata": { "dir": "input", "type": "i32" },
                    "we": { "dir": "input", "type": "i1" },
                    "rdata": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "load": {
                      "ii": 1,
                      "d": 1,
                      "bind": {
                        "addr": "operand1",
                        "rdata": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        self.assertEqual([port.name for port in uglir_design.inputs].count("A_rdata"), 1)
        self.assertEqual([port.name for port in uglir_design.outputs].count("A_addr"), 1)
        self.assertEqual(len([resource for resource in uglir_design.resources if resource.kind == "port" and resource.id == "A"]), 1)
        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("sel_mr0_addr_n", "state_q == 2 ? SRC_I : state_q == 8 ? SRC_J : ZERO"), assigns)
        self.assertIn(("A_addr", "mr0_addr_q"), assigns)
        addr_mux = next(mux for mux in uglir_design.muxes if mux.name == "mx_mr0_addr_n")
        self.assertEqual([(case.key, case.source) for case in addr_mux.cases], [("ZERO", "const_i32_0_n"), ("SRC_I", "i"), ("SRC_J", "j")])
        rendered = format_uglir(uglir_design)
        self.assertIn("assign A_addr = mr0_addr_q", rendered)

    def test_lower_fsm_to_uglir_resolves_unrolled_issue_occurrences_for_fu_input_muxes(self) -> None:
        fsm_design = parse_uhir(
            """
            design mul_once
            stage fsm
            schedule kind=control_steps
            input  x : i32
            input  y : i32
            output result : i32
            resources {
              fu mul0 : MUL
              reg r_i32_0 : i32
            }
            controller C0 encoding=binary protocol=req_resp completion_order=in_order overlap=true region=proc_mul_once {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=0
              state T0 code=1
              state T1 code=2
              state DONE code=3
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mul0<-v1@0]
              emit T1 latch=[r_i32_0]
              emit DONE resp_valid=true
            }

            region proc_mul_once kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = mul x, y : i32 class=MUL ii=1 delay=2 start=0 end=1 bind=mul0
              node v2 = ret v1 class=CTRL ii=0 delay=0 start=2 end=2
              node v3 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              steps [0:2]
              latency 3
              value v1 -> r_i32_0 live=[2:2]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "MUL": {
                  "kind": "combinational",
                  "ports": {
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "mul": {
                      "ii": 1,
                      "d": 2,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)

        assigns = {(assign.target, assign.expr) for assign in uglir_design.assigns}
        self.assertIn(("sel_mul0_a_n", "state_q == 1 ? SRC_X : ZERO"), assigns)
        self.assertIn(("sel_mul0_b_n", "state_q == 1 ? SRC_Y : ZERO"), assigns)
        mux_a = next(mux for mux in uglir_design.muxes if mux.name == "mx_mul0_a_n")
        mux_b = next(mux for mux in uglir_design.muxes if mux.name == "mx_mul0_b_n")
        self.assertEqual([(case.key, case.source) for case in mux_a.cases], [("ZERO", "const_i32_0_n"), ("SRC_X", "x")])
        self.assertEqual([(case.key, case.source) for case in mux_b.cases], [("ZERO", "const_i32_0_n"), ("SRC_Y", "y")])

    def test_lower_fsm_to_uglir_rejects_same_state_competing_accesses_to_one_memory_interface(self) -> None:
        fsm_design = parse_uhir(
            """
            design read_conflict
            stage fsm
            schedule kind=control_steps
            input  A : memref<i32>
            input  i : i32
            input  j : i32
            output result : i32
            resources {
              fu mr0 : MEM
            }
            controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_read_conflict {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=1
              state T0 code=2
              state DONE code=4
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mr0<-v1, mr0<-v2]
              emit DONE resp_valid=true
            }

            region proc_read_conflict kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = load A, i : i32 class=MEM ii=1 delay=1 start=0 end=0 bind=mr0
              node v2 = load A, j : i32 class=MEM ii=1 delay=1 start=0 end=0 bind=mr0
              node v3 = ret 0:i32 class=CTRL ii=0 delay=0 start=1 end=1
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
              edge data v0 -> v1
              edge data v0 -> v2
              edge data v1 -> v3
              edge data v2 -> v3
              edge data v3 -> v4
              steps [0:1]
              latency 2
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "MEM": {
                  "kind": "memory",
                  "ports": {
                    "addr": { "dir": "input", "type": "i32" },
                    "wdata": { "dir": "input", "type": "i32" },
                    "we": { "dir": "input", "type": "i1" },
                    "rdata": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "load": {
                      "ii": 1,
                      "d": 1,
                      "bind": {
                        "addr": "operand1",
                        "rdata": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        with self.assertRaises(ValueError) as raised:
            lower_fsm_to_uglir(fsm_design, component_library=component_library)

        self.assertIn("one access per memory interface per FSMD state", str(raised.exception))
        self.assertIn("memory 'A'", str(raised.exception))
        self.assertIn("v1, v2", str(raised.exception))

    def test_parse_uglir_roundtrips(self) -> None:
        design = parse_uglir(
            """
            design glue
            stage uglir
            input  clk : clock
            input  rst : i1
            input  req_valid : i1
            input  x : i32
            output req_ready : i1
            output resp_valid : i1
            output result : i32
            resources {
              reg state : u3
              net next_state : u3
              net ewms0_go : i1
              net ewms0_y : i32
              reg r_i32_0 : i32
              net latch_r_i32_0 : i1
              net sel_r_i32_0 : ctrl
              inst ewms0 : EWMS
              mux mx_r_i32_0 : i32
            }

            assign req_ready = state == 0

            ewms0.go(ewms0_go)

            mux mx_r_i32_0 : i32 sel=sel_r_i32_0 {
              hold -> r_i32_0
              ewms0_y -> ewms0_y
            }

            seq clk {
              if rst {
                state <= 0
              } else {
                state <= next_state
                if latch_r_i32_0 {
                  r_i32_0 <= mx_r_i32_0
                }
              }
            }
            """
        )

        self.assertEqual(design.stage, "uglir")
        self.assertEqual(len(design.resources), 9)
        self.assertEqual(design.attachments[0].instance, "ewms0")
        self.assertEqual(design.muxes[0].name, "mx_r_i32_0")
        self.assertEqual(design.seq_blocks[0].updates[1].enable, "latch_r_i32_0")
        rendered = format_uglir(design)
        self.assertIn("stage uglir", rendered)
        self.assertIn("inst ewms0 : EWMS", rendered)
        self.assertIn("ewms0.go(ewms0_go)", rendered)


class UGLIRSyntaxTests(unittest.TestCase):
    """Coverage for textual µglIR parsing and formatting."""

    def test_parse_uglir_with_address_map(self) -> None:
        design = parse_uglir(
            """
            design wrapped
            stage uglir
            input  clk : clock
            input  idx : u2
            output wb_ack_o : i1
            output A_rdata : i32
            const  WB_REG_CONTROL_STATUS = WB_BASE_ADDR + 32'h0000_0000 : u32

            address_map wishbone {
              register control_status offset=32'h0000_0000 access=rw symbol=WB_REG_CONTROL_STATUS
              register x offset=32'h0000_0100 access=rw symbol=WB_REG_IN_X type=i32
              memory A offset=32'h0000_1000 span=32'h0000_0010 access=rw symbol=WB_MEM_A_BASE word_t=i32 depth=4
            }

            resources {
              net wb_req_n : i1
              mem A_mem_q : i32[4]
            }

            assign wb_ack_o = wb_req_n
            assign A_rdata = A_mem_q[idx]

            seq clk {
              A_mem_q[idx] <= 0:i32
            }
            """
        )

        self.assertEqual(len(design.address_maps), 1)
        self.assertEqual(design.address_maps[0].name, "wishbone")
        self.assertEqual(design.address_maps[0].entries[0].kind, "register")
        self.assertEqual(design.address_maps[0].entries[1].attributes["type"], "i32")
        self.assertEqual(design.address_maps[0].entries[2].attributes["depth"], 4)
        self.assertEqual(design.resources[1].kind, "mem")
        self.assertEqual(design.seq_blocks[0].updates[0].target, "A_mem_q[idx]")
        self.assertIn("address_map wishbone {", format_uglir(design))

    def test_parse_and_format_uglir_preserve_instance_target(self) -> None:
        design = parse_uglir(
            """
            design wrapped
            stage uglir
            input  clk : clock
            resources {
              inst A_mem_inst : RM_IHPSG13_1P_64x64_c2_bm_bist<W=64,DEPTH=64> RM_IHPSG13_1P_64x64_c2_bm_bist<word_t=i64,word_len=64>
            }
            seq clk {
            }
            """
        )

        self.assertEqual(
            [(resource.kind, resource.id, resource.value, resource.target) for resource in design.resources],
            [
                (
                    "inst",
                    "A_mem_inst",
                    "RM_IHPSG13_1P_64x64_c2_bm_bist<W=64,DEPTH=64>",
                    "RM_IHPSG13_1P_64x64_c2_bm_bist<word_t=i64,word_len=64>",
                )
            ],
        )
        rendered = format_uglir(design)
        self.assertIn(
            "inst A_mem_inst : RM_IHPSG13_1P_64x64_c2_bm_bist<W=64,DEPTH=64> RM_IHPSG13_1P_64x64_c2_bm_bist<word_t=i64,word_len=64>",
            rendered,
        )
        reparsed = parse_uglir(rendered)
        self.assertEqual(
            [(resource.kind, resource.id, resource.value, resource.target) for resource in reparsed.resources],
            [(resource.kind, resource.id, resource.value, resource.target) for resource in design.resources],
        )

    def test_parse_uglir_accepts_expression_sequential_enable(self) -> None:
        design = parse_uglir(
            """
            design gated
            stage uglir
            input  clk : clock
            input  a : i1
            input  b : i1
            output y : i1
            resources {
              reg r_q : i1
            }

            assign y = r_q

            seq clk {
              r_q <= false
              if a && b {
                r_q <= true
              }
            }
            """
        )

        self.assertEqual(design.seq_blocks[0].updates[1].enable, "a && b")

    def test_parse_uglir_accepts_constants_in_sequential_enable_expressions(self) -> None:
        design = parse_uglir(
            """
            design wrapped
            stage uglir
            input  clk : clock
            input  obi_accept_n : i1
            input  obi_we_i : i1
            input  obi_addr_i : u32
            output y : i1
            const  OBI_REG_IN_APACKED_4XI8_0 = 32'h0000_0100 : u32
            resources {
              reg r_q : i1
            }

            assign y = r_q

            seq clk {
              if false {
                r_q <= false
              } else {
                if (obi_accept_n && obi_we_i) && (obi_addr_i == OBI_REG_IN_APACKED_4XI8_0) {
                  r_q <= true
                }
              }
            }
            """
        )

        self.assertEqual(
            design.seq_blocks[0].updates[0].enable,
            "(obi_accept_n && obi_we_i) && (obi_addr_i == OBI_REG_IN_APACKED_4XI8_0)",
        )

    def test_parse_uglir_accepts_multiline_parenthesized_expressions(self) -> None:
        design = parse_uglir(
            """
            design wrapped
            stage uglir
            input  clk : clock
            input  a : i1
            input  b : i1
            output y : i1
            resources {
              reg r_q : i1
              net n_n : i1
            }

            assign n_n = (
              a &&
              b
            )

            assign y = (
              n_n
            )

            seq clk {
              r_q <= false
              if (
                a &&
                b
              ) {
                r_q <= (
                  n_n
                )
              }
            }
            """
        )

        self.assertEqual(design.assigns[0].expr, "a && b")
        self.assertEqual(design.assigns[1].expr, "n_n")
        self.assertEqual(design.seq_blocks[0].updates[1].enable, "a && b")
        self.assertEqual(design.seq_blocks[0].updates[1].value, "n_n")

    def test_format_uglir_wraps_long_expressions_in_parentheses(self) -> None:
        design = parse_uglir(
            """
            design wrapped
            stage uglir
            input  clk : clock
            input  req_valid : i1
            input  req_ready : i1
            output y : i1
            resources {
              reg r_q : i1
              net n_n : i1
            }

            assign n_n = req_valid && req_ready && req_valid && req_ready && req_valid && req_ready && req_valid && req_ready && req_valid && req_ready
            assign y = n_n

            seq clk {
              r_q <= req_valid && req_ready && req_valid && req_ready && req_valid && req_ready && req_valid && req_ready && req_valid && req_ready
            }
            """
        )

        rendered = format_uglir(design)
        self.assertIn("assign n_n = (", rendered)
        self.assertIn("r_q <= (", rendered)

    def test_lower_fsm_to_uglir_updates_parent_phi_carries_for_canonicalized_unrolled_loop_backedges(self) -> None:
        uglir_design = _lower_unrolled_dot4_relu_to_uglir()

        seq_block = uglir_design.seq_blocks[0]
        phi_sum_update = next(update for update in seq_block.updates if update.target == "phi_sum_1_q")
        phi_i_update = next(update for update in seq_block.updates if update.target == "phi_i_1_q")

        self.assertIsNotNone(phi_sum_update.enable)
        self.assertIsNotNone(phi_i_update.enable)
        self.assertNotEqual(phi_sum_update.enable, "req_fire_n")
        self.assertNotEqual(phi_i_update.enable, "req_fire_n")
        self.assertIn("state_q ==", phi_sum_update.enable)
        self.assertIn("state_q ==", phi_i_update.enable)
        self.assertGreaterEqual(phi_sum_update.enable.count("state_q =="), 2)
        self.assertGreaterEqual(phi_i_update.enable.count("state_q =="), 2)
        self.assertIn("inl_mac_0_t1_0__u1_n", phi_sum_update.value)
        self.assertNotIn("t4_0_n", phi_i_update.value)

        select_assign = next(assign for assign in uglir_design.assigns if assign.target == "sel_r_i32_0_n")
        self.assertIn("state_q ==", select_assign.expr)
        self.assertGreaterEqual(select_assign.expr.count("state_q =="), 4)

        result_assign = next(assign for assign in uglir_design.assigns if assign.target == "result")
        self.assertIn("phi_sum_1_q", result_assign.expr)

    def test_lower_fsm_to_uglir_updates_header_phi_from_completed_latch_phi_for_flattened_unroll(self) -> None:
        uglir_design = _lower_unrolled_example_to_uglir(
            "dot4_relu",
            factor=2,
            component_library=_generic_component_library(),
            flatten=True,
        )

        seq_block = uglir_design.seq_blocks[0]
        phi_i_update = next(update for update in seq_block.updates if update.target == "phi_i_1_q")
        phi_sum_update = next(update for update in seq_block.updates if update.target == "phi_sum_1_q")

        self.assertIn("state_q == 6", phi_i_update.enable)
        self.assertIn("state_q == 12", phi_i_update.enable)
        self.assertNotIn("state_q == 2", phi_i_update.enable)
        self.assertNotIn("state_q == 8", phi_i_update.enable)
        self.assertIn("t4_0__u1_n", phi_i_update.value)
        self.assertNotIn("t4_0_n", phi_i_update.value)

        self.assertIn("state_q == 6", phi_sum_update.enable)
        self.assertIn("state_q == 12", phi_sum_update.enable)
        self.assertNotIn("state_q == 5", phi_sum_update.enable)
        self.assertNotIn("state_q == 11", phi_sum_update.enable)
        self.assertIn("inl_mac_0_t1_0__u1_n", phi_sum_update.value)
        self.assertNotIn("inl_mac_0_t1_0_n", phi_sum_update.value)

    def test_lower_fsm_to_uglir_handles_n_way_phi_carries_for_unroll_by_4(self) -> None:
        uglir_design = _lower_unrolled_example_to_uglir("dot4_i8_i32_relu_packed", factor=4)

        seq_block = uglir_design.seq_blocks[0]
        phi_sum_latch_update = next(update for update in seq_block.updates if update.target == "phi_sum_1_latch_q")
        phi_sum_update = next(update for update in seq_block.updates if update.target == "phi_sum_1_q")

        self.assertIn("inl_mac_0_t3_0__u1_n", phi_sum_latch_update.value)
        self.assertIn("inl_mac_0_t3_0__u2_n", phi_sum_latch_update.value)
        self.assertIn("inl_mac_0_t3_0__u3_n", phi_sum_latch_update.value)
        self.assertGreaterEqual(phi_sum_latch_update.enable.count("state_q =="), 3)

        self.assertIn("inl_mac_0_t3_0__u3_n", phi_sum_update.value)
        self.assertGreaterEqual(phi_sum_update.enable.count("state_q =="), 1)

        select_assign = next(assign for assign in uglir_design.assigns if assign.target == "sel_r_i32_0_n")
        self.assertGreaterEqual(select_assign.expr.count("state_q =="), 4)

    def test_lower_fsm_to_uglir_lowers_non_loop_merge_phi_as_branch_select(self) -> None:
        uglir_design = _lower_unrolled_example_to_uglir(
            "dot4_i8_i32_relu_packed",
            factor=4,
            optimize=False,
            cleanup_after_unroll=False,
        )

        self.assertFalse(any(resource.id == "phi_sum_4_q" for resource in uglir_design.resources))

        result_assign = next(assign for assign in uglir_design.assigns if assign.target == "result")
        self.assertIn("? 0:i32 :", result_assign.expr)
        self.assertIn("phi_sum_1_q", result_assign.expr)

        seq_block = uglir_design.seq_blocks[0]
        phi_sum_update = next(update for update in seq_block.updates if update.target == "phi_sum_1_q")
        self.assertIn("(req_fire_n) ? C_0", phi_sum_update.value)
        self.assertIn("r_i32_0_q", phi_sum_update.value)
        self.assertNotIn("r_i32_1_q", phi_sum_update.value)

    def test_lower_fsm_to_uglir_threads_late_mov_aliases_through_semantic_nets(self) -> None:
        uglir_design = _lower_unrolled_example_to_uglir(
            "dot4_i8_i32_relu_packed",
            factor=4,
            optimize=False,
            cleanup_after_unroll=False,
        )

        assign_by_target = {assign.target: assign.expr for assign in uglir_design.assigns}

        self.assertEqual(assign_by_target["sum_2_n"], "state_q == 13 ? ewms0_y_n : r_i32_0_q")
        self.assertEqual(assign_by_target["inl_mac_0_c_0__u1_n"], "state_q == 14 ? ewms0_y_n : r_i32_0_q")
        self.assertEqual(assign_by_target["sum_2__u1_n"], "state_q == 19 ? ewms0_y_n : r_i32_0_q")
        self.assertEqual(assign_by_target["inl_mac_0_c_0__u2_n"], "state_q == 20 ? ewms0_y_n : r_i32_0_q")
        self.assertEqual(assign_by_target["sum_2__u2_n"], "state_q == 25 ? ewms0_y_n : r_i32_0_q")
        self.assertEqual(assign_by_target["inl_mac_0_c_0__u3_n"], "state_q == 26 ? ewms0_y_n : r_i32_0_q")

    def test_lower_fsm_to_uglir_stabilizes_scalar_late_aliases_after_capture_without_opt_cleanup(self) -> None:
        uglir_design = _lower_example_to_uglir(
            "dot4_relu",
            optimize=False,
        )

        assign_by_target = {assign.target: assign.expr for assign in uglir_design.assigns}

        self.assertEqual(assign_by_target["inl_mac_0_t1_0_n"], "state_q == 6 ? ewms0_y_n : r_i32_0_q")
        self.assertEqual(assign_by_target["inl_mac_0_acc_1_n"], "state_q == 7 ? ewms0_y_n : r_i32_0_q")
        self.assertEqual(assign_by_target["t3_0_n"], "state_q == 8 ? ewms0_y_n : r_i32_0_q")
        self.assertEqual(assign_by_target["sum_2_n"], "state_q == 9 ? ewms0_y_n : r_i32_0_q")

    def test_lower_fsm_to_uglir_uses_final_loop_carried_sum_for_post_loop_compare_without_opt_cleanup(self) -> None:
        uglir_design = _lower_example_to_uglir(
            "dot4_i8_i32_relu_packed",
            optimize=False,
        )

        assign_by_target = {assign.target: assign.expr for assign in uglir_design.assigns}

        self.assertIn("state_q == 13 ? SRC_SUM_2", assign_by_target["sel_r_i32_0_n"])
        self.assertNotIn("state_q == 13 ? SRC_INL_MAC_0_T3_0", assign_by_target["sel_r_i32_0_n"])

    def test_fsm_global_live_steps_do_not_reexpand_already_scheduled_static_loop_bodies(self) -> None:
        fsm_design = _lower_example_to_fsm(
            "dot4_i8_i32_relu_packed",
            optimize=False,
        )
        producer_region = _producer_region(fsm_design, "sum_2")

        self.assertIsNotNone(producer_region)
        self.assertEqual(sorted(_value_global_live_starts(fsm_design, "sum_2")), [12])
        self.assertEqual(
            sorted(_producer_global_capture_steps(fsm_design, producer_region.id, "sum_2", None)),
            [12],
        )

    def test_lower_fsm_to_uglir_avoids_phantom_post_loop_phi_carry_updates_without_opt_cleanup(self) -> None:
        uglir_design = _lower_example_to_uglir(
            "dot4_i8_i32_relu_packed",
            optimize=False,
        )

        seq_block = uglir_design.seq_blocks[0]
        phi_sum_update = next(update for update in seq_block.updates if update.target == "phi_sum_1_q")
        r_i32_0_update = next(update for update in seq_block.updates if update.target == "r_i32_0_q")

        self.assertNotIn("state_q == 72", phi_sum_update.value)
        self.assertNotIn("state_q == 72", phi_sum_update.enable or "")
        self.assertNotIn("state_q == 59", r_i32_0_update.enable or "")
        self.assertNotIn("state_q == 71", r_i32_0_update.enable or "")
        self.assertNotIn("state_q == 72", r_i32_0_update.enable or "")
        self.assertIn("state_q == 12) ? sum_2_n", phi_sum_update.value)
        self.assertNotIn("state_q == 12) ? r_i32_0_q", phi_sum_update.value)

    def test_lower_fsm_to_uglir_uses_entry_value_for_phi_carry_initialization_without_opt_cleanup(self) -> None:
        uglir_design = _lower_example_to_uglir(
            "dot4_i8_i32_relu_packed",
            optimize=False,
        )

        seq_block = uglir_design.seq_blocks[0]
        phi_sum_update = next(update for update in seq_block.updates if update.target == "phi_sum_1_q")

        self.assertIn("(req_fire_n) ? C_0", phi_sum_update.value)
        self.assertNotIn("(req_fire_n) ? sum_0_n", phi_sum_update.value)

    def test_lower_fsm_to_uglir_bypasses_staging_registers_for_immediate_mac_operands_without_opt_cleanup(self) -> None:
        uglir_design = _lower_example_to_uglir(
            "dot4_i8_i32_relu_packed",
            optimize=False,
        )

        rendered = format_uglir(uglir_design)

        self.assertIn("SRC_INL_MAC_0_T1_0", rendered)
        self.assertIn("SRC_INL_MAC_0_T2_0", rendered)
        self.assertNotIn("sel_gen_mul_pseudopipe_ii1d30_a_n = (\n  state_q == 2 ? SRC_PHI_I_1 : state_q == 8 ? SRC_R_I16_0", rendered)
        self.assertNotIn("sel_gen_mul_pseudopipe_ii1d30_b_n = (\n  state_q == 2 ? CONST_8_I8 : state_q == 8 ? SRC_R_I16_1", rendered)

    def test_lower_fsm_to_uglir_keeps_pipelined_result_live_for_next_cycle_held_input_consumer(self) -> None:
        fsm_design = parse_uhir(
            """
            design mul_alias
            stage fsm
            schedule kind=control_steps
            input  x : i32
            output result : i32
            resources {
              fu mul0 : MUL
              fu alu0 : ALU
              reg r_i32_0 : i32
              reg r_i32_1 : i32
            }
            controller C0 encoding=binary protocol=req_resp completion_order=in_order overlap=true region=proc_mul_alias {
              input  req_valid : i1
              input  resp_ready : i1
              output req_ready : i1
              output resp_valid : i1
              state IDLE code=0
              state T0 code=1
              state T1 code=2
              state T2 code=3
              state T3 code=4
              state DONE code=5
              transition IDLE -> T0 when=req_valid && req_ready
              transition T0 -> T1
              transition T1 -> T2
              transition T2 -> T3
              transition T3 -> DONE
              transition DONE -> IDLE when=resp_valid && resp_ready
              emit IDLE req_ready=true
              emit T0 issue=[mul0<-v1]
              emit T2 issue=[alu0<-v2]
              emit DONE resp_valid=true
            }

            region proc_mul_alias kind=procedure {
              node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
              node v1 = mul x, 2:i32 : i32 class=MUL ii=1 delay=2 start=0 end=1 bind=mul0
              node v2 = add v1, 0:i32 : i32 class=ALU ii=1 delay=1 start=2 end=2 bind=alu0
              node v3 = ret v2 class=CTRL ii=0 delay=0 start=3 end=3
              node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4
              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
              steps [0:3]
              latency 4
              value v1 -> r_i32_0 live=[2:2]
              value v2 -> r_i32_1 live=[3:3]
            }
            """
        )
        component_library = json.loads(
            """
            {
              "components": {
                "ALU": {
                  "kind": "combinational",
                  "ports": {
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "op": { "dir": "input", "type": "u5" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "add": {
                      "ii": 1,
                      "d": 1,
                      "opcode": 0,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                },
                "MUL": {
                  "kind": "pipelined",
                  "ports": {
                    "clk": { "dir": "input", "type": "clock" },
                    "rst": { "dir": "input", "type": "reset", "active": "hi" },
                    "a": { "dir": "input", "type": "i32" },
                    "b": { "dir": "input", "type": "i32" },
                    "y": { "dir": "output", "type": "i32" }
                  },
                  "supports": {
                    "mul": {
                      "ii": 1,
                      "d": 2,
                      "bind": {
                        "a": "operand0",
                        "b": "operand1",
                        "y": "result"
                      }
                    }
                  }
                }
              }
            }
            """
        )["components"]

        uglir_design = lower_fsm_to_uglir(fsm_design, component_library=component_library)
        assign_by_target = {assign.target: assign.expr for assign in uglir_design.assigns}

        self.assertEqual(assign_by_target["v1_n"], "state_q == 2 ? mul0_y_n : r_i32_0_q")
        self.assertEqual(assign_by_target["sel_alu0_a_n"], "state_q == 3 ? SRC_R_I32_0 : ZERO")
