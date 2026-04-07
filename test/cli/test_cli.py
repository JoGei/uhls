from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from uhls.cli import main
from uhls.middleend.uir import COMPACT_OPCODE_LABELS


class CLITests(unittest.TestCase):
    """End-to-end coverage for the µhLS CLI surface."""

    def test_parse_verify_cfg_dfg_and_cdfg_commands_round_trip(self) -> None:
        source = """
        int32_t add1(int32_t x) {
            return x + 1;
        }
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uc_path = root / "add1.uc"
            uir_path = root / "add1.uir"
            uc_path.write_text(source, encoding="utf-8")

            self.assertEqual(main(["parse", str(uc_path), "-o", str(uir_path)]), 0)
            self.assertTrue(uir_path.read_text(encoding="utf-8").startswith("func add1"))

            verify_out = io.StringIO()
            with redirect_stdout(verify_out):
                self.assertEqual(main(["lint", str(uir_path)]), 0)
            self.assertEqual(verify_out.getvalue().strip(), "ok")

            cfg_out = io.StringIO()
            with redirect_stdout(cfg_out):
                self.assertEqual(main(["cfg", str(uir_path), "--dot"]), 0)
            self.assertIn('digraph "add1"', cfg_out.getvalue())

            dfg_out = io.StringIO()
            with redirect_stdout(dfg_out):
                self.assertEqual(main(["dfg", str(uir_path), "--dot"]), 0)
            self.assertIn('digraph "add1.dfg"', dfg_out.getvalue())
            self.assertIn('"entry:0" [label="t0_0:i32 = add x_0, 1:i32"];', dfg_out.getvalue())

            compact_dfg_out = io.StringIO()
            with redirect_stdout(compact_dfg_out):
                self.assertEqual(main(["dfg", str(uir_path), "--dot", "--compact"]), 0)
            self.assertIn('"entry:0" [label="+"];', compact_dfg_out.getvalue())
            self.assertIn('"entry:0" -> "entry:term" [label="t0_0"];', compact_dfg_out.getvalue())

            cdfg_out = io.StringIO()
            with redirect_stdout(cdfg_out):
                self.assertEqual(main(["cdfg", str(uir_path)]), 0)
            self.assertIn("func add1: blocks=1 control_edges=0", cdfg_out.getvalue())
            self.assertIn("block entry: dfg_nodes=2 dfg_edges=1", cdfg_out.getvalue())

            cdfg_dot_out = io.StringIO()
            with redirect_stdout(cdfg_dot_out):
                self.assertEqual(main(["cdfg", str(uir_path), "--dot"]), 0)
            self.assertIn('digraph "add1.cdfg"', cdfg_dot_out.getvalue())
            self.assertIn('subgraph "cluster_entry"', cdfg_dot_out.getvalue())

            compact_cdfg_dot_out = io.StringIO()
            with redirect_stdout(compact_cdfg_dot_out):
                self.assertEqual(main(["cdfg", str(uir_path), "--dot", "--compact"]), 0)
            self.assertIn('"entry:0" [label="+"];', compact_cdfg_dot_out.getvalue())
            self.assertIn('"entry:0" -> "entry:term" [label="t0_0"', compact_cdfg_dot_out.getvalue())

    def test_seq_command_lowers_uir_to_seq_uhir_and_renders_dot(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    y:i32 = add x, 1
    ret y
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uir_path = root / "add1.uir"
            seq_path = root / "add1.seq.uhir"
            dot_path = root / "add1.dot"
            uir_path.write_text(uir, encoding="utf-8")

            self.assertEqual(main(["seq", str(uir_path), "-o", str(seq_path)]), 0)
            seq_text = seq_path.read_text(encoding="utf-8")
            self.assertIn("stage seq", seq_text)
            self.assertIn("region proc_add1 kind=procedure {", seq_text)
            self.assertIn("nop", seq_text)

            self.assertEqual(main(["seq", str(seq_path), "--dot", "-o", str(dot_path)]), 0)
            dot_text = dot_path.read_text(encoding="utf-8")
            self.assertIn('digraph "add1.seq"', dot_text)
            self.assertIn('cluster_proc_add1', dot_text)

    def test_gopt_command_runs_builtin_infer_static_on_seq_uhir(self) -> None:
        uir = """func dot4(A:i32[], B:i32[]) -> i32

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
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uir_path = root / "dot4.uir"
            seq_path = root / "dot4.seq.uhir"
            opt_path = root / "dot4.gopt.seq.uhir"
            uir_path.write_text(uir, encoding="utf-8")

            self.assertEqual(main(["seq", str(uir_path), "-o", str(seq_path)]), 0)
            self.assertEqual(main(["gopt", str(seq_path), "-p", "infer_loops,translate_loop_dialect,infer_static", "-o", str(opt_path)]), 0)

            optimized_text = opt_path.read_text(encoding="utf-8")
            self.assertIn("static_trip_count=4", optimized_text)

    def test_gopt_command_runs_infer_loops_on_seq_uhir(self) -> None:
        uir = """func dot4(A:i32[], B:i32[]) -> i32

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
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uir_path = root / "dot4.uir"
            seq_path = root / "dot4.seq.uhir"
            opt_path = root / "dot4.loop.seq.uhir"
            uir_path.write_text(uir, encoding="utf-8")

            self.assertEqual(main(["seq", str(uir_path), "-o", str(seq_path)]), 0)
            self.assertEqual(main(["gopt", str(seq_path), "-p", "infer_loops,translate_loop_dialect", "-o", str(opt_path)]), 0)

            optimized_text = opt_path.read_text(encoding="utf-8")
            self.assertIn("loop_id=L0", optimized_text)
            self.assertIn("loop_header=true", optimized_text)
            self.assertIn("loop_backedge=true", optimized_text)

    def test_gopt_command_accepts_external_pass(self) -> None:
        seq = """design add1
stage seq

region proc_add1 kind=procedure {
  node v0 = nop role=source
  node v1 = add x, 1 : i32
  node v2 = ret v1
  node v3 = nop role=sink

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        plugin = """
from copy import deepcopy

class ExternalGOptPass:
    def __init__(self, pass_args=()):
        self.pass_args = tuple(pass_args)

    def run(self, ir):
        out = deepcopy(ir)
        out.regions[0].nodes[1].attributes["tag"] = self.pass_args[0] if self.pass_args else "external"
        return out
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_path = root / "add1.seq.uhir"
            plugin_path = root / "external_gopt.py"
            out_path = root / "add1.gopt.uhir"
            seq_path.write_text(seq, encoding="utf-8")
            plugin_path.write_text(plugin, encoding="utf-8")

            self.assertEqual(
                main(
                    [
                        "gopt",
                        str(seq_path),
                        "-p",
                        f"{plugin_path}:ExternalGOptPass",
                        "--pass-arg",
                        "hello",
                        "-o",
                        str(out_path),
                    ]
                ),
                0,
            )

            optimized_text = out_path.read_text(encoding="utf-8")
            self.assertIn("tag=hello", optimized_text)

    def test_gopt_command_rejects_dot_input_with_clear_error(self) -> None:
        dot = 'digraph "dot4.seq" {\n}\n'
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dot_path = root / "dot4.seq.uhir.dot"
            dot_path.write_text(dot, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["gopt", str(dot_path), "-p", "infer_static"]), 1)
            self.assertIn("expects one µhIR input file, not Graphviz DOT", stderr.getvalue())

    def test_gopt_command_supports_dot_output(self) -> None:
        seq = """design add1
stage seq

region proc_add1 kind=procedure {
  node v0 = nop role=source
  node v1 = add x, 1 : i32
  node v2 = ret v1
  node v3 = nop role=sink

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_path = root / "add1.seq.uhir"
            dot_path = root / "add1.gopt.dot"
            seq_path.write_text(seq, encoding="utf-8")

            self.assertEqual(main(["gopt", str(seq_path), "-p", "infer_loops,translate_loop_dialect,infer_static", "--dot", "-o", str(dot_path)]), 0)

            dot_text = dot_path.read_text(encoding="utf-8")
            self.assertIn('digraph "add1.seq"', dot_text)
            self.assertIn('cluster_proc_add1', dot_text)

    def test_gopt_command_simplifies_static_loop_control(self) -> None:
        uir = """func dot4(A:i32[], B:i32[]) -> i32

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
    t3_0:i32 = mul t1_0, t2_0
    inl_mac_0_t1_0:i32 = add sum_1, t3_0
    t4_0:i32 = add i_1, 1:i32
    br for_header_1

block for_exit_4:
    ret sum_1
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uir_path = root / "dot4.uir"
            seq_path = root / "dot4.seq.uhir"
            opt_path = root / "dot4.opt.seq.uhir"
            uir_path.write_text(uir, encoding="utf-8")

            self.assertEqual(main(["seq", str(uir_path), "-o", str(seq_path)]), 0)
            self.assertEqual(
                main(
                    [
                        "gopt",
                        str(seq_path),
                        "-p",
                        "infer_loops,translate_loop_dialect,infer_static,simplify_static_control",
                        "-o",
                        str(opt_path),
                    ]
                ),
                0,
            )

            optimized_text = opt_path.read_text(encoding="utf-8")
            self.assertIn("static_trip_count=4", optimized_text)
            self.assertNotIn("lt i_1, 4:i32", optimized_text)
            self.assertIn("node v", optimized_text)
            self.assertIn("true_child=", optimized_text)
            self.assertIn("false_child=", optimized_text)

    def test_gopt_command_projects_bind_input_back_to_seq(self) -> None:
        bind = """design add1
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
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bind_path = root / "add1.bind.uhir"
            out_path = root / "add1.gopt.uhir"
            bind_path.write_text(bind, encoding="utf-8")

            self.assertEqual(main(["gopt", str(bind_path), "-p", "infer_static", "-o", str(out_path)]), 0)

            optimized_text = out_path.read_text(encoding="utf-8")
            self.assertIn("stage seq", optimized_text)
            self.assertNotIn("resources {", optimized_text)
            self.assertNotIn("class=EWMS", optimized_text)
            self.assertNotIn("start=0", optimized_text)
            self.assertNotIn("value t0_0 ->", optimized_text)

    def test_alloc_command_lowers_seq_to_alloc_uhir_and_renders_dot(self) -> None:
        seq = """design add1
stage seq

region proc_add1 kind=procedure {
  node v0 = nop role=source
  node v1 = add x, 1 : i32
  node v2 = ret v1
  node v3 = nop role=sink

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        graph = """
{
  "functional_units": ["fu_generic", "fu_fast_add"],
  "operations": ["add", "sub", "mul", "div", "mod", "and", "or", "xor", "shl", "shr", "eq", "ne", "lt", "le", "gt", "ge", "neg", "not", "mov", "const", "load", "store", "phi", "call", "print", "param", "br", "cbr", "ret"],
  "edges": [
    ["fu_generic", "add", {"ii": 2, "d": 2}],
    ["fu_generic", "sub", {"ii": 1, "d": 1}],
    ["fu_generic", "mul", {"ii": 1, "d": 1}],
    ["fu_generic", "div", {"ii": 1, "d": 1}],
    ["fu_generic", "mod", {"ii": 1, "d": 1}],
    ["fu_generic", "and", {"ii": 1, "d": 1}],
    ["fu_generic", "or", {"ii": 1, "d": 1}],
    ["fu_generic", "xor", {"ii": 1, "d": 1}],
    ["fu_generic", "shl", {"ii": 1, "d": 1}],
    ["fu_generic", "shr", {"ii": 1, "d": 1}],
    ["fu_generic", "eq", {"ii": 1, "d": 1}],
    ["fu_generic", "ne", {"ii": 1, "d": 1}],
    ["fu_generic", "lt", {"ii": 1, "d": 1}],
    ["fu_generic", "le", {"ii": 1, "d": 1}],
    ["fu_generic", "gt", {"ii": 1, "d": 1}],
    ["fu_generic", "ge", {"ii": 1, "d": 1}],
    ["fu_generic", "neg", {"ii": 1, "d": 1}],
    ["fu_generic", "not", {"ii": 1, "d": 1}],
    ["fu_generic", "mov", {"ii": 1, "d": 1}],
    ["fu_generic", "const", {"ii": 1, "d": 1}],
    ["fu_generic", "load", {"ii": 1, "d": 1}],
    ["fu_generic", "store", {"ii": 1, "d": 1}],
    ["fu_generic", "phi", {"ii": 1, "d": 1}],
    ["fu_generic", "call", {"ii": 1, "d": 1}],
    ["fu_generic", "print", {"ii": 1, "d": 1}],
    ["fu_generic", "param", {"ii": 1, "d": 1}],
    ["fu_generic", "br", {"ii": 1, "d": 1}],
    ["fu_generic", "cbr", {"ii": 1, "d": 1}],
    ["fu_generic", "ret", {"ii": 1, "d": 1}],
    ["fu_fast_add", "add", {"ii": 1, "d": 1}]
  ]
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_path = root / "add1.seq.uhir"
            graph_path = root / "exec.json"
            alloc_path = root / "add1.alloc.uhir"
            dot_path = root / "add1.alloc.dot"
            seq_path.write_text(seq, encoding="utf-8")
            graph_path.write_text(graph, encoding="utf-8")

            self.assertEqual(
                main(["alloc", str(seq_path), "-exg", str(graph_path), "-o", str(alloc_path)]),
                0,
            )
            alloc_text = alloc_path.read_text(encoding="utf-8")
            self.assertIn("stage alloc", alloc_text)
            self.assertIn("class=FU_FAST_ADD", alloc_text)
            self.assertIn("ii=1", alloc_text)
            self.assertIn("delay=1", alloc_text)
            self.assertIn("region executability_graph kind=executability {", alloc_text)
            self.assertIn("node FU_FAST_ADD = fu partition=fu", alloc_text)
            self.assertIn("node CTRL = fu partition=fu", alloc_text)
            self.assertIn("edge exg FU_FAST_ADD -- add ii=1 d=1", alloc_text)
            self.assertIn("edge exg CTRL -- nop ii=0 d=0", alloc_text)

            self.assertEqual(
                main(["alloc", str(seq_path), "-exg", str(graph_path), "--dot", "-o", str(dot_path)]),
                0,
            )
            dot_text = dot_path.read_text(encoding="utf-8")
            self.assertIn('digraph "add1.alloc"', dot_text)
            self.assertIn('cluster_proc_add1', dot_text)

    def test_alloc_command_supports_min_ii_algorithm(self) -> None:
        seq = """design add1
stage seq

region proc_add1 kind=procedure {
  node v0 = nop role=source
  node v1 = add x, 1 : i32
  node v2 = ret v1
  node v3 = nop role=sink

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        graph = """
{
  "functional_units": ["fu_fast_ii", "fu_fast_delay"],
  "operations": ["add", "sub", "mul", "div", "mod", "and", "or", "xor", "shl", "shr", "eq", "ne", "lt", "le", "gt", "ge", "neg", "not", "mov", "const", "load", "store", "phi", "call", "print", "param", "br", "cbr", "ret"],
  "edges": [
    ["fu_fast_delay", "add", {"ii": 2, "d": 2}],
    ["fu_fast_ii", "add", {"ii": 1, "d": 3}],
    ["fu_fast_delay", "sub", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "mul", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "div", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "mod", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "and", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "or", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "xor", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "shl", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "shr", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "eq", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "ne", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "lt", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "le", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "gt", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "ge", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "neg", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "not", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "mov", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "const", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "load", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "store", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "phi", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "call", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "print", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "param", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "br", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "cbr", {"ii": 1, "d": 1}],
    ["fu_fast_delay", "ret", {"ii": 1, "d": 1}]
  ]
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_path = root / "add1.seq.uhir"
            graph_path = root / "exec.json"
            alloc_path = root / "add1.alloc.uhir"
            seq_path.write_text(seq, encoding="utf-8")
            graph_path.write_text(graph, encoding="utf-8")

            self.assertEqual(main(["alloc", str(seq_path), "-exg", str(graph_path), "--algo", "min_ii", "-o", str(alloc_path)]), 0)

            alloc_text = alloc_path.read_text(encoding="utf-8")
            self.assertIn("class=FU_FAST_II", alloc_text)
            self.assertIn("ii=1", alloc_text)
            self.assertIn("delay=3", alloc_text)

    def test_alloc_command_accepts_component_library_json(self) -> None:
        seq = """design add1
stage seq

region proc_add1 kind=procedure {
  node v0 = nop role=source
  node v1 = add x, 1 : i32
  node v2 = ret v1
  node v3 = nop role=sink

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        library = """
{
  "components": {
    "ALU": {
      "kind": "combinational",
      "ports": {
        "a": { "dir": "input", "type": "i32" },
        "b": { "dir": "input", "type": "i32" },
        "op": { "dir": "input", "type": "u4" },
        "y": { "dir": "output", "type": "i32" }
      },
      "supports": {
        "add": { "ii": 1, "d": 1, "opcode": 0 },
        "and": { "ii": 1, "d": 1, "opcode": 1 },
        "eq": { "ii": 1, "d": 1, "opcode": 2 },
        "ge": { "ii": 1, "d": 1, "opcode": 3 },
        "gt": { "ii": 1, "d": 1, "opcode": 4 },
        "le": { "ii": 1, "d": 1, "opcode": 5 },
        "lt": { "ii": 1, "d": 1, "opcode": 6 },
        "mov": { "ii": 1, "d": 1, "opcode": 7 },
        "ne": { "ii": 1, "d": 1, "opcode": 8 },
        "neg": { "ii": 1, "d": 1, "opcode": 9 },
        "not": { "ii": 1, "d": 1, "opcode": 10 },
        "or": { "ii": 1, "d": 1, "opcode": 11 },
        "shl": { "ii": 1, "d": 1, "opcode": 12 },
        "shr": { "ii": 1, "d": 1, "opcode": 13 },
        "sub": { "ii": 1, "d": 1, "opcode": 14 },
        "xor": { "ii": 1, "d": 1, "opcode": 15 }
      }
    },
    "MUL": {
      "kind": "combinational",
      "ports": {
        "a": { "dir": "input", "type": "i32" },
        "b": { "dir": "input", "type": "i32" },
        "y": { "dir": "output", "type": "i32" }
      },
      "supports": {
        "mul": { "ii": 1, "d": 2 }
      }
    },
    "DIV": {
      "kind": "combinational",
      "ports": {
        "a": { "dir": "input", "type": "i32" },
        "b": { "dir": "input", "type": "i32" },
        "op": { "dir": "input", "type": "u1" },
        "y": { "dir": "output", "type": "i32" }
      },
      "supports": {
        "div": { "ii": 1, "d": 3, "opcode": 0 },
        "mod": { "ii": 1, "d": 3, "opcode": 1 }
      }
    },
    "MEM": {
      "kind": "memory",
      "ports": {
        "addr": { "dir": "input", "type": "i32" },
        "wdata": { "dir": "input", "type": "i32" },
        "we": { "dir": "input", "type": "i1" },
        "rdata": { "dir": "output", "type": "i32" }
      },
      "supports": {
        "load": { "ii": 1, "d": 1, "mode": "read" },
        "store": { "ii": 1, "d": 1, "mode": "write" }
      }
    }
  }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_path = root / "add1.seq.uhir"
            library_path = root / "ressources.json"
            alloc_path = root / "add1.alloc.uhir"
            seq_path.write_text(seq, encoding="utf-8")
            library_path.write_text(library, encoding="utf-8")

            self.assertEqual(main(["alloc", str(seq_path), "-exg", str(library_path), "-o", str(alloc_path)]), 0)

            alloc_text = alloc_path.read_text(encoding="utf-8")
            self.assertIn("stage alloc", alloc_text)
            self.assertIn("node v1 = add x, 1 : i32 class=ALU ii=1 delay=1", alloc_text)
            self.assertIn("node v2 = ret v1 class=CTRL ii=0 delay=0", alloc_text)

    def test_alloc_command_rejects_invalid_component_parameter_schema(self) -> None:
        seq = """design add1
stage seq

region proc_add1 kind=procedure {
  node v0 = nop role=source
  node v1 = add x, 1 : i32
  node v2 = ret v1
  node v3 = nop role=sink

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        library = """
{
  "components": {
    "MEM": {
      "kind": "memory",
      "parameters": {
        "word_len": { "kind": "integer" }
      },
      "ports": {
        "addr": { "dir": "input", "type": "i32" },
        "rdata": { "dir": "output", "type": "i32" }
      },
      "supports": {
        "load": { "ii": 1, "d": 1 }
      }
    }
  }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_path = root / "add1.seq.uhir"
            library_path = root / "ressources.json"
            seq_path.write_text(seq, encoding="utf-8")
            library_path.write_text(library, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["alloc", str(seq_path), "-exg", str(library_path)]), 1)
            self.assertIn("parameter 'word_len' must use kind=type|int|bool|string", stderr.getvalue())

    def test_sched_command_lowers_alloc_to_sched_with_builtin_asap(self) -> None:
        alloc = """design add1
stage alloc

region proc_add1 kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0
  node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1
  node v2 = ret v1 class=CTRL ii=0 delay=0
  node v3 = nop role=sink class=CTRL ii=0 delay=0

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alloc_path = root / "add1.alloc.uhir"
            sched_path = root / "add1.sched.uhir"
            alloc_path.write_text(alloc, encoding="utf-8")

            self.assertEqual(main(["sched", str(alloc_path), "--algo", "asap", "-o", str(sched_path)]), 0)

            sched_text = sched_path.read_text(encoding="utf-8")
            self.assertIn("stage sched", sched_text)
            self.assertIn("schedule kind=hierarchical", sched_text)
            self.assertIn("node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1 start=0 end=0", sched_text)
            self.assertIn("node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1", sched_text)
            self.assertIn("steps [0:1]", sched_text)
            self.assertIn("latency 2", sched_text)

    def test_sched_command_requires_algo(self) -> None:
        alloc = """design add1
stage alloc

region proc_add1 kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0
  node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1
  node v2 = ret v1 class=CTRL ii=0 delay=0
  node v3 = nop role=sink class=CTRL ii=0 delay=0

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alloc_path = root / "add1.alloc.uhir"
            alloc_path.write_text(alloc, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["sched", str(alloc_path)]), 2)
            self.assertIn("Missing option '--algo'", stderr.getvalue())

    def test_sched_command_accepts_external_flat_sgu_scheduler(self) -> None:
        alloc = """design add1
stage alloc

region proc_add1 kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0
  node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1
  node v2 = ret v1 class=CTRL ii=0 delay=0
  node v3 = nop role=sink class=CTRL ii=0 delay=0

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        plugin = """
from uhls.backend.hls import SGUScheduleResult

def external_sched(region):
    starts = {}
    for node in region.nodes:
        if node.opcode == "add":
            starts[node.id] = (4, 4)
        elif node.opcode == "ret":
            starts[node.id] = (6, 6)
        else:
            starts[node.id] = (0, 0)
    return SGUScheduleResult(region.id, starts, latency=7)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alloc_path = root / "add1.alloc.uhir"
            plugin_path = root / "external_sched.py"
            sched_path = root / "add1.sched.uhir"
            alloc_path.write_text(alloc, encoding="utf-8")
            plugin_path.write_text(plugin, encoding="utf-8")

            self.assertEqual(main(["sched", str(alloc_path), "--algo", f"{plugin_path}:external_sched", "-o", str(sched_path)]), 0)

            sched_text = sched_path.read_text(encoding="utf-8")
            self.assertIn("stage sched", sched_text)
            self.assertIn("node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1 start=4 end=4", sched_text)
            self.assertIn("node v2 = ret v1 class=CTRL ii=0 delay=0 start=6 end=6", sched_text)
            self.assertIn("steps [4:6]", sched_text)
            self.assertIn("latency 7", sched_text)

    def test_sched_command_accepts_external_alap_scheduler_with_asap_slack(self) -> None:
        alloc = """design add1
stage alloc

region proc_add1 kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0
  node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1
  node v2 = ret v1 class=CTRL ii=0 delay=0
  node v3 = nop role=sink class=CTRL ii=0 delay=0

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alloc_path = root / "add1.alloc.uhir"
            sched_path = root / "add1.sched.uhir"
            alloc_path.write_text(alloc, encoding="utf-8")

            alap_module = Path.cwd() / "alap_scheduler.py"
            self.assertEqual(
                main(
                    [
                        "sched",
                        str(alloc_path),
                        "--algo",
                        f"{alap_module}:ALAPScheduler",
                        "--sgu_latency_max",
                        "asap+1",
                        "-o",
                        str(sched_path),
                    ]
                ),
                0,
            )

            sched_text = sched_path.read_text(encoding="utf-8")
            self.assertIn("stage sched", sched_text)
            self.assertIn("node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1 start=2 end=2", sched_text)
            self.assertIn("node v2 = ret v1 class=CTRL ii=0 delay=0 start=3 end=3", sched_text)
            self.assertIn("latency 3", sched_text)

    def test_sched_command_accepts_builtin_alap_scheduler_with_asap_slack(self) -> None:
        alloc = """design add1
stage alloc

region proc_add1 kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0
  node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1
  node v2 = ret v1 class=CTRL ii=0 delay=0
  node v3 = nop role=sink class=CTRL ii=0 delay=0

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alloc_path = root / "add1.alloc.uhir"
            sched_path = root / "add1.sched.uhir"
            alloc_path.write_text(alloc, encoding="utf-8")

            self.assertEqual(
                main(
                    [
                        "sched",
                        str(alloc_path),
                        "--algo",
                        "alap",
                        "--sgu_latency_max",
                        "asap+1",
                        "-o",
                        str(sched_path),
                    ]
                ),
                0,
            )

            sched_text = sched_path.read_text(encoding="utf-8")
            self.assertIn("stage sched", sched_text)
            self.assertIn("node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1 start=2 end=2", sched_text)
            self.assertIn("node v2 = ret v1 class=CTRL ii=0 delay=0 start=3 end=3", sched_text)
            self.assertIn("latency 3", sched_text)

    def test_sched_command_external_alap_requires_sgu_latency_max(self) -> None:
        alloc = """design add1
stage alloc

region proc_add1 kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0
  node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1
  node v2 = ret v1 class=CTRL ii=0 delay=0
  node v3 = nop role=sink class=CTRL ii=0 delay=0

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alloc_path = root / "add1.alloc.uhir"
            alloc_path.write_text(alloc, encoding="utf-8")

            alap_module = Path.cwd() / "alap_scheduler.py"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["sched", str(alloc_path), "--algo", f"{alap_module}:ALAPScheduler"]), 1)
            self.assertIn("ALAPScheduler requires --sgu_latency_max", stderr.getvalue())

    def test_sched_command_external_alap_rejects_infeasible_latency_max(self) -> None:
        alloc = """design add1
stage alloc

region proc_add1 kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0
  node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1
  node v2 = ret v1 class=CTRL ii=0 delay=0
  node v3 = nop role=sink class=CTRL ii=0 delay=0

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alloc_path = root / "add1.alloc.uhir"
            alloc_path.write_text(alloc, encoding="utf-8")

            alap_module = Path.cwd() / "alap_scheduler.py"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(
                    main(
                        [
                            "sched",
                            str(alloc_path),
                            "--algo",
                            f"{alap_module}:ALAPScheduler",
                            "--sgu_latency_max",
                            "proc_add1:1",
                        ]
                    ),
                    1,
                )
            self.assertIn("minimum feasible latency is 2", stderr.getvalue())

    def test_sched_command_external_alap_handles_longer_dependency_chain_with_asap_latency(self) -> None:
        alloc = """design chain
stage alloc

region proc_chain kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0
  node v1 = load A, i : i32 class=FU_GENERIC ii=1 delay=1
  node v2 = mul v1, c : i32 class=FU_GENERIC ii=1 delay=1
  node v3 = add acc, v2 : i32 class=FU_GENERIC ii=1 delay=1
  node v4 = nop role=sink class=CTRL ii=0 delay=0

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
  edge data v3 -> v4
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alloc_path = root / "chain.alloc.uhir"
            sched_path = root / "chain.sched.uhir"
            alloc_path.write_text(alloc, encoding="utf-8")

            alap_module = Path.cwd() / "alap_scheduler.py"
            self.assertEqual(
                main(
                    [
                        "sched",
                        str(alloc_path),
                        "--algo",
                        f"{alap_module}:ALAPScheduler",
                        "--sgu_latency_max",
                        "asap",
                        "-o",
                        str(sched_path),
                    ]
                ),
                0,
            )

            sched_text = sched_path.read_text(encoding="utf-8")
            self.assertIn("node v1 = load A, i : i32 class=FU_GENERIC ii=1 delay=1 start=0 end=0", sched_text)
            self.assertIn("node v2 = mul v1, c : i32 class=FU_GENERIC ii=1 delay=1 start=1 end=1", sched_text)
            self.assertIn("node v3 = add acc, v2 : i32 class=FU_GENERIC ii=1 delay=1 start=2 end=2", sched_text)
            self.assertIn("latency 3", sched_text)

    def test_alloc_command_can_emit_dummy_executability_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dummy_path = root / "exec.json"

            self.assertEqual(main(["alloc", "--gen_dummy_exg", "-o", str(dummy_path)]), 0)

            dummy_text = dummy_path.read_text(encoding="utf-8")
            self.assertIn('"functional_units": [', dummy_text)
            self.assertIn('"EWMS"', dummy_text)
            self.assertIn('"edges": [', dummy_text)
            self.assertIn('"add"', dummy_text)
            self.assertIn('"ii": 1', dummy_text)
            self.assertIn('"d": 1', dummy_text)

    def test_alloc_command_can_pretty_print_executability_graph_without_input(self) -> None:
        graph = """
{
  "functional_units": ["fu_generic"],
  "operations": ["add"],
  "edges": [
    ["fu_generic", "add", {"ii": 1, "d": 1}]
  ]
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_path = root / "exec.json"
            graph_path.write_text(graph, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["alloc", "-exg", str(graph_path)]), 0)

            rendered = stdout.getvalue()
            self.assertIn("executability_graph {", rendered)
            self.assertIn("fu FU_GENERIC", rendered)
            self.assertIn("op add", rendered)
            self.assertIn("edge FU_GENERIC -- add ii=1 d=1", rendered)

    def test_alloc_command_accepts_exg_uhir_collateral(self) -> None:
        seq = """design add1
stage seq

region proc_add1 kind=procedure {
  node src = nop role=source
  node y = add x, 1 : i32
  node ret0 = ret y

  edge data y -> ret0
}
"""
        graph = (
            """
design demo_exg
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
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_path = root / "add1.seq.uhir"
            graph_path = root / "exec.uhir"
            alloc_path = root / "add1.alloc.uhir"
            seq_path.write_text(seq, encoding="utf-8")
            graph_path.write_text(graph, encoding="utf-8")

            self.assertEqual(main(["alloc", str(seq_path), "-exg", str(graph_path), "-o", str(alloc_path)]), 0)

            alloc_text = alloc_path.read_text(encoding="utf-8")
            self.assertIn("stage alloc", alloc_text)
            self.assertIn("class=FU_FAST_ADD", alloc_text)
            self.assertIn("ii=1", alloc_text)
            self.assertIn("delay=1", alloc_text)
            self.assertIn("region executability_graph kind=executability {", alloc_text)

    def test_alloc_command_can_render_executability_graph_dot_without_input(self) -> None:
        graph = """
{
  "functional_units": ["fu_generic"],
  "operations": ["add"],
  "edges": [
    ["fu_generic", "add", {"ii": 1, "d": 1}]
  ]
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_path = root / "exec.json"
            graph_path.write_text(graph, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["alloc", "-exg", str(graph_path), "--dot"]), 0)

            rendered = stdout.getvalue()
            self.assertIn('graph "executability_graph"', rendered)
            self.assertIn('"fu:FU_GENERIC"', rendered)
            self.assertIn('"op:add"', rendered)
            self.assertIn('label="ii=1, d=1"', rendered)

    def test_seq_command_accepts_top_for_multi_function_modules(self) -> None:
        uir = """func helper(x:i32) -> i32

block entry:
    y:i32 = add x, 1
    ret y

func top(x:i32) -> i32

block entry:
    z:i32 = call helper(x)
    ret z
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["seq", str(path), "--top", "top"]), 0)

            rendered = stdout.getvalue()
            self.assertIn("design top", rendered)
            self.assertIn("region proc_top kind=procedure {", rendered)
            self.assertIn("region proc_helper kind=procedure {", rendered)

    def test_seq_command_can_render_dot_directly_from_uir_input(self) -> None:
        uir = """func dot4(A:i32[], B:i32[]) -> i32

block entry:
    sum_0:i32 = const 0:i32
    ret sum_0
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dot4.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["seq", str(path), "--top", "dot4", "--dot"]), 0)

            rendered = stdout.getvalue()
            self.assertIn('digraph "dot4.seq"', rendered)
            self.assertIn('cluster_proc_dot4', rendered)

    def test_seq_command_supports_compact_dot_labels(self) -> None:
        uir = """func dot4(A:i32[], B:i32[]) -> i32

block entry:
    sum_0:i32 = const 0:i32
    ret sum_0
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dot4.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["seq", str(path), "--top", "dot4", "--dot", "--compact"]), 0)

            rendered = stdout.getvalue()
            self.assertIn('"v0" [label="v0: nop"', rendered)
            self.assertIn('"v1" [label="v1: const"', rendered)
            self.assertIn('"v2" [label="v2: ret"', rendered)
            self.assertNotIn("ret sum_0", rendered)

    def test_dfg_command_can_select_one_block(self) -> None:
        uir = """func select(sel:i1, a:i32, b:i32) -> i32

block entry:
    cbr sel, then_blk, else_blk

block then_blk:
    x:i32 = add a, 1
    ret x

block else_blk:
    y:i32 = sub b, 1
    ret y
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "select.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["dfg", str(path), "--block", "then_blk", "--dot"]), 0)

            rendered = stdout.getvalue()
            self.assertIn('digraph "select.then_blk.dfg"', rendered)
            self.assertIn("node [shape=ellipse];", rendered)
            self.assertNotIn('cluster_else_blk', rendered)

    def test_dfg_dot_without_function_merges_multiple_function_graphs(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    y:i32 = add x, 1
    ret y

func sub1(x:i32) -> i32

block entry:
    y:i32 = sub x, 1
    ret y
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["dfg", str(path), "--dot"]), 0)

            rendered = stdout.getvalue()
            self.assertEqual(rendered.count("digraph "), 1)
            self.assertIn('digraph "module.dfg"', rendered)
            self.assertIn('subgraph "cluster_add1_dfg"', rendered)
            self.assertIn('subgraph "cluster_sub1_dfg"', rendered)
            self.assertIn('"add1:entry:0"', rendered)
            self.assertIn('"sub1:entry:0"', rendered)
            self.assertIn("node [shape=ellipse];", rendered)

    def test_cfg_dot_without_function_merges_multiple_function_cfgs(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    y:i32 = add x, 1
    ret y

func sub1(x:i32) -> i32

block entry:
    y:i32 = sub x, 1
    ret y
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["cfg", str(path), "--dot"]), 0)

            rendered = stdout.getvalue()
            self.assertEqual(rendered.count("digraph "), 1)
            self.assertIn('digraph "module.cfg"', rendered)
            self.assertIn('subgraph "cluster_add1"', rendered)
            self.assertIn('subgraph "cluster_sub1"', rendered)
            self.assertIn('"add1:entry"', rendered)
            self.assertIn('"sub1:entry"', rendered)

    def test_cdfg_without_function_merges_multiple_function_graphs(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    y:i32 = add x, 1
    ret y

func sub1(x:i32) -> i32

block entry:
    y:i32 = sub x, 1
    ret y
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["cdfg", str(path), "--dot"]), 0)

            rendered = stdout.getvalue()
            self.assertEqual(rendered.count("digraph "), 1)
            self.assertIn('digraph "module.cdfg"', rendered)
            self.assertIn('subgraph "cluster_add1_cdfg"', rendered)
            self.assertIn('subgraph "cluster_sub1_cdfg"', rendered)
            self.assertIn('"add1:entry:0"', rendered)
            self.assertIn('"sub1:entry:0"', rendered)

    def test_run_command_executes_uir_and_prints_return_value(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    t0:i32 = add x, 1
    ret t0
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "add1.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["run", str(path), "--arg", "x=7"]), 0)
            self.assertEqual(stdout.getvalue().strip(), "8")

    def test_run_command_accepts_array_parameters_via_arg(self) -> None:
        uir = """func first_sum(A:i32[], B:i32[]) -> i32

block entry:
    idx:i32 = const 0:i32
    a0:i32 = load A[idx]
    b0:i32 = load B[idx]
    s0:i32 = add a0, b0
    ret s0
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "first_sum.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(["run", str(path), "--arg", "A=[1,2,3,4]", "--arg", "B=[4,3,2,1]"]),
                    0,
                )
            self.assertEqual(stdout.getvalue().strip(), "5")

    def test_run_command_reports_missing_array_arguments_cleanly(self) -> None:
        uir = """func dot4(A:i32[], B:i32[]) -> i32

block entry:
    x:i32 = const 0:i32
    ret x
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dot4.uir"
            path.write_text(uir, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["run", str(path), "--function", "dot4"]), 1)

            rendered = stderr.getvalue()
            self.assertIn("missing array arguments for function 'dot4': A, B", rendered)
            self.assertIn("pass --arg name=[v1,v2,...][:type]", rendered)
            self.assertNotIn("Traceback", rendered)

    def test_run_command_reports_invalid_call_targets_cleanly(self) -> None:
        uir = """func caller(x:i32) -> i32

block entry:
    y:i32 = call missing(x)
    ret y
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "caller.uir"
            path.write_text(uir, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["run", str(path), "--function", "caller", "--arg", "x=7"]), 1)

            rendered = stderr.getvalue()
            self.assertIn("call to unknown function 'missing'", rendered)
            self.assertNotIn("Traceback", rendered)

    def test_lint_accepts_local_arrays_and_print(self) -> None:
        uir = """func dot4(A:i32[], B:i32[]) -> i32

block entry:
    x:i32 = const 0:i32
    ret x

func main() -> i32

local main$A[4]:i32
local main$B[4]:i32

block entry:
    store main$A[0:i32], 1:i32
    store main$B[0:i32], 2:i32
    res_0:i32 = call dot4(main$A, main$B)
    print "res=%d", res_0
    ret 0:i32
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tb.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["lint", str(path)]), 0)
            self.assertEqual(stdout.getvalue().strip(), "ok")

    def test_lint_accepts_uhir(self) -> None:
        uhir = """design add1
stage seq

region proc_add1 kind=procedure {
  node v0 = nop role=source
  node v1 = add x, 1 : i32
  node v2 = ret v1
  node v3 = nop role=sink

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "add1.seq.uhir"
            path.write_text(uhir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["lint", str(path)]), 0)
            self.assertEqual(stdout.getvalue().strip(), "ok")

    def test_lint_accepts_uglir(self) -> None:
        uglir = """design add1
stage uglir
input  clk : clock
input  rst : i1
output req_ready : i1
resources {
  reg state_q : u1
  net next_state_n : u1
}
assign req_ready = true
seq clk {
  state_q <= next_state_n
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "add1.uglir"
            path.write_text(uglir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["lint", str(path)]), 0)
            self.assertEqual(stdout.getvalue().strip(), "ok")

    def test_lint_accepts_component_library_json(self) -> None:
        library = {
            "components": {
                "MEM": {
                    "kind": "memory",
                    "parameters": {
                        "word_t": {"kind": "type"},
                        "word_len": {"kind": "int"},
                    },
                    "ports": {
                        "addr": {"dir": "input", "type": "i32"},
                        "rdata": {"dir": "output", "type": "i32"},
                    },
                    "supports": {
                        "load": {"ii": 1, "d": 1},
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ressources.json"
            path.write_text(json.dumps(library), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["lint", str(path)]), 0)
            self.assertEqual(stdout.getvalue().strip(), "ok")

    def test_opt_command_runs_constprop_pipeline(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    t0:i32 = const 1:i32
    t1:i32 = add t0, 1
    ret t1
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "add1.uir"
            out_path = Path(tmpdir) / "add1_opt.uir"
            path.write_text(uir, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["opt", str(path), "-p", "constprop,dce", "-o", str(out_path)]), 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("ret 2:i32", out_path.read_text(encoding="utf-8"))

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["opt", str(path), "-p", "constprop,dce"]), 0)
            self.assertIn("ret 2:i32", stdout.getvalue())

    def test_opt_command_runs_implemented_cfg_transform_passes(self) -> None:
        uir = """func cleanup() -> i32

block entry:
    br jump

block jump:
    br exit

block exit:
    x:i32 = const 7:i32
    ret x
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cleanup.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["opt", str(path), "-p", "simplify_cfg"]), 0)

            rendered = stdout.getvalue()
            self.assertIn("func cleanup", rendered)
            self.assertNotIn("block jump:", rendered)

    def test_opt_command_inline_calls_warns_for_missing_requested_callee(self) -> None:
        uir = """func foo(x:i32) -> i32

block entry:
    y:i32 = add x, 1
    ret y

func caller(x:i32) -> i32

block entry:
    y:i32 = call foo(x)
    ret y
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "caller.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main(["opt", str(path), "-p", "inline_calls", "--pass-arg", "missing"]), 0)

            self.assertIn("func caller", stdout.getvalue())
            self.assertIn(
                "warning: inline_calls: requested callee 'missing' was not found in the translation unit",
                stderr.getvalue(),
            )

    def test_opt_command_loads_external_pass_from_python_file(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    t0:i32 = add x, 1
    ret t0
"""
        plugin = """
from copy import deepcopy

def external_mark(ir):
    result = deepcopy(ir)
    result.name = "external_opt_ran"
    return result
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "add1.uir"
            plugin_path = root / "external_pass.py"
            path.write_text(uir, encoding="utf-8")
            plugin_path.write_text(plugin, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["opt", str(path), "-p", f"{plugin_path}:external_mark"]), 0)

            rendered = stdout.getvalue()
            self.assertIn("module external_opt_ran", rendered)
            self.assertIn("func add1", rendered)

    def test_opt_command_loads_external_pass_module_that_uses_dataclass(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    t0:i32 = add x, 1
    ret t0
"""
        plugin = """
from dataclasses import dataclass
from copy import deepcopy

@dataclass(frozen=True)
class Marker:
    name: str = "external_dataclass_opt"

def external_mark(ir):
    result = deepcopy(ir)
    result.name = Marker().name
    return result
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "add1.uir"
            plugin_path = root / "external_pass.py"
            path.write_text(uir, encoding="utf-8")
            plugin_path.write_text(plugin, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["opt", str(path), "-p", f"{plugin_path}:external_mark"]), 0)

            self.assertIn("module external_dataclass_opt", stdout.getvalue())

    def test_opt_command_forwards_shared_pass_args_to_external_callable(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    t0:i32 = add x, 1
    ret t0
"""
        plugin = """
from copy import deepcopy

def external_mark(ir, context, pass_args):
    result = deepcopy(ir)
    result.name = ",".join(pass_args)
    context.data["seen_pass_args"] = list(pass_args)
    return result
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "add1.uir"
            plugin_path = root / "external_pass.py"
            path.write_text(uir, encoding="utf-8")
            plugin_path.write_text(plugin, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "opt",
                            str(path),
                            "-p",
                            f"{plugin_path}:external_mark",
                            "--pass-arg",
                            "caller",
                            "--pass-arg",
                            "4",
                        ]
                    ),
                    0,
                )

            self.assertIn("module caller,4", stdout.getvalue())

    def test_opt_command_forwards_shared_pass_args_to_external_pass_class_constructor(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    t0:i32 = add x, 1
    ret t0
"""
        plugin = """
from copy import deepcopy

class ExternalPass:
    def __init__(self, pass_args):
        self.name = "external_ctor"
        self._pass_args = pass_args

    def run(self, ir, context):
        result = deepcopy(ir)
        result.name = ":".join(self._pass_args)
        return result
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "add1.uir"
            plugin_path = root / "external_pass.py"
            path.write_text(uir, encoding="utf-8")
            plugin_path.write_text(plugin, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "opt",
                            str(path),
                            "-p",
                            f"{plugin_path}:ExternalPass",
                            "--pass-arg",
                            "accum",
                            "--pass-arg",
                            "8",
                        ]
                    ),
                    0,
                )

            self.assertIn("module accum:8", stdout.getvalue())

    def test_opt_command_loads_external_zero_arg_pass_factory(self) -> None:
        uir = """func cleanup() -> i32

block entry:
    br jump

block jump:
    br exit

block exit:
    x:i32 = const 7:i32
    ret x
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cleanup.uir"
            path.write_text(uir, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "opt",
                            str(path),
                            "-p",
                            "./src/uhls/passes/opt/simplify_cfg.py:SimplifyCFGPass",
                        ]
                    ),
                    0,
                )

            rendered = stdout.getvalue()
            self.assertIn("func cleanup", rendered)
            self.assertNotIn("block jump:", rendered)

    def test_opt_command_reports_external_analysis_helpers_cleanly(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    t0:i32 = add x, 1
    ret t0
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "add1.uir"
            path.write_text(uir, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(
                    main(["opt", str(path), "-p", "./src/uhls/passes/analyze/cfg.py:control_flow"]),
                    1,
                )

            rendered = stderr.getvalue()
            self.assertIn("optimization pipeline failed", rendered)
            self.assertIn("has no attribute 'blocks'", rendered)
            self.assertNotIn("Traceback", rendered)

    def test_bind_command_lowers_sched_to_bind(self) -> None:
        sched = """design add_pair
stage sched
schedule kind=control_steps

region proc_add_pair kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = add a, b : i32 class=FU_ADD ii=1 delay=1 start=0 end=0
  node v2 = add c, d : i32 class=FU_ADD ii=1 delay=1 start=0 end=0
  node v3 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
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
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sched_path = root / "add_pair.sched.uhir"
            bind_path = root / "add_pair.bind.uhir"
            sched_path.write_text(sched, encoding="utf-8")

            self.assertEqual(main(["bind", str(sched_path), "--algo", "left_edge", "-o", str(bind_path)]), 0)

            bind_text = bind_path.read_text(encoding="utf-8")
            self.assertIn("stage bind", bind_text)
            self.assertIn("fu fu_add0 : FU_ADD", bind_text)
            self.assertIn("fu fu_add1 : FU_ADD", bind_text)
            self.assertIn("reg r_i32_0 : i32", bind_text)
            self.assertIn("reg r_i32_1 : i32", bind_text)
            self.assertIn("value v1 -> r_i32_0 live=[1:1]", bind_text)
            self.assertIn("node v1 = add a, b : i32 class=FU_ADD ii=1 delay=1 start=0 end=0 bind=fu_add0", bind_text)

    def test_bind_command_prefers_mapped_value_ids_in_emitted_bindings(self) -> None:
        sched = """design mapped_values
stage sched
schedule kind=control_steps

region proc_mapped_values kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = add a, b : i32 class=FU_ADD ii=1 delay=1 start=0 end=0
  node v2 = add v1, c : i32 class=FU_ADD ii=1 delay=1 start=1 end=1
  node v3 = ret v2 class=CTRL ii=0 delay=0 start=2 end=2
  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
  edge data v3 -> v4

  map v1 <- t1_0
  map v2 <- t2_0
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sched_path = root / "mapped_values.sched.uhir"
            bind_path = root / "mapped_values.bind.uhir"
            sched_path.write_text(sched, encoding="utf-8")

            self.assertEqual(main(["bind", str(sched_path), "--algo", "left_edge", "-o", str(bind_path)]), 0)

            bind_text = bind_path.read_text(encoding="utf-8")
            self.assertIn("value t1_0 -> r_i32_0 live=[1:1]", bind_text)
            self.assertIn("value t2_0 -> r_i32_0 live=[2:2]", bind_text)

    def test_bind_command_supports_compat_for_symbolic_sched(self) -> None:
        sched = """design dyn_compat
stage sched
schedule kind=hierarchical

region proc_dyn_compat kind=procedure {
  region_ref proc_callee
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = call x child=proc_callee class=CTRL ii=II delay=symb_delay_v1 start=0 end=symb_delay_v1 - 1 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done
  node v2 = add a, b : i32 class=FU_ADD ii=1 delay=1 start=max(0, symb_delay_v1) end=max(0, symb_delay_v1) + 1 - 1
  node v3 = add v2, c : i32 class=FU_ADD ii=1 delay=1 start=max(0, max(0, symb_delay_v1) + 1) end=max(0, max(0, symb_delay_v1) + 1) + 1 - 1
  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
  edge data v3 -> v4
}

region proc_callee kind=procedure {
  node c0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node c1 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
  edge data c0 -> c1
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sched_path = root / "dyn_compat.sched.uhir"
            bind_path = root / "dyn_compat.bind.uhir"
            sched_path.write_text(sched, encoding="utf-8")

            self.assertEqual(main(["bind", str(sched_path), "--algo", "compat", "-o", str(bind_path)]), 0)

            bind_text = bind_path.read_text(encoding="utf-8")
            self.assertIn("stage bind", bind_text)
            self.assertIn("fu fu_add0 : FU_ADD", bind_text)
            self.assertNotIn("reg ", bind_text)
            self.assertIn("node v2 = add a, b : i32 class=FU_ADD ii=1 delay=1 start=symb_delay_v1 end=symb_delay_v1 bind=fu_add0", bind_text)
            self.assertIn("node v3 = add v2, c : i32 class=FU_ADD ii=1 delay=1 start=symb_delay_v1 + 1 end=symb_delay_v1 + 1 bind=fu_add0", bind_text)

    def test_bind_command_rejects_flatten_with_compat(self) -> None:
        sched = """design dyn_compat
stage sched
schedule kind=hierarchical

region proc_dyn_compat kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = add x, y : i32 class=FU_ADD ii=1 delay=1 start=max(0, symb_delay_v4) end=max(0, symb_delay_v4) + 1 - 1
  node v2 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
  edge data v0 -> v1
  edge data v1 -> v2
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sched_path = root / "dyn_compat.sched.uhir"
            sched_path.write_text(sched, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["bind", str(sched_path), "--algo", "compat", "--flatten"]), 1)
            self.assertIn("does not support --flatten", stderr.getvalue())

    def test_fsm_command_lowers_bind_to_fsm(self) -> None:
        bind = """design add1
stage bind
schedule kind=control_steps
resources {
  fu ewms0 : EWMS
  reg r_i32_0 : i32
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
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bind_path = root / "add1.bind.uhir"
            fsm_path = root / "add1.fsm.uhir"
            bind_path.write_text(bind, encoding="utf-8")

            self.assertEqual(main(["fsm", str(bind_path), "--encoding", "one_hot", "-o", str(fsm_path)]), 0)

            fsm_text = fsm_path.read_text(encoding="utf-8")
            self.assertIn("stage fsm", fsm_text)
            self.assertIn("controller C0 encoding=one_hot protocol=req_resp completion_order=in_order overlap=true region=proc_add1 {", fsm_text)
            self.assertIn("input  req_valid : i1", fsm_text)
            self.assertIn("output resp_valid : i1", fsm_text)
            self.assertIn("state IDLE code=1", fsm_text)
            self.assertIn("transition IDLE -> T0 when=req_valid && req_ready", fsm_text)
            self.assertIn("emit T0 issue=[ewms0<-v1]", fsm_text)
            self.assertIn("emit T1 latch=[r_i32_0]", fsm_text)

    def test_fsm_command_supports_dot_output(self) -> None:
        bind = """design add1
stage bind
schedule kind=control_steps
resources {
  fu ewms0 : EWMS
  reg r_i32_0 : i32
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
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bind_path = root / "add1.bind.uhir"
            dot_path = root / "add1.fsm.dot"
            bind_path.write_text(bind, encoding="utf-8")

            self.assertEqual(main(["fsm", str(bind_path), "--encoding", "one_hot", "--dot", "-o", str(dot_path)]), 0)

            dot_text = dot_path.read_text(encoding="utf-8")
            self.assertIn('digraph "add1.fsm"', dot_text)
            self.assertIn('"C0:IDLE"', dot_text)
            self.assertIn('"C0:IDLE" -> "C0:T0"', dot_text)
            self.assertIn('"C0:IDLE" [label="IDLE\ncode=1\nreq_ready=true"', dot_text)
            self.assertIn('"C0:T0" [label="T0\ncode=2\nissue=[ewms0<-v1]"', dot_text)

    def test_fsm_command_emits_mux_select_actions(self) -> None:
        bind = """design acc
stage bind
schedule kind=control_steps
resources {
  fu ewms0 : EWMS
  reg r_acc : i32
}

region proc_acc kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = add acc, x : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=ewms0
  node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3

  steps [0:1]
  latency 2
  value v1 -> r_acc live=[1:1]
  mux mx0 : input=[r_acc, r_next] output=r_acc sel=state
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bind_path = root / "acc.bind.uhir"
            fsm_path = root / "acc.fsm.uhir"
            bind_path.write_text(bind, encoding="utf-8")

            self.assertEqual(main(["fsm", str(bind_path), "--encoding", "binary", "-o", str(fsm_path)]), 0)

            fsm_text = fsm_path.read_text(encoding="utf-8")
            self.assertIn("emit T1 latch=[r_acc] select=[mx0<-state]", fsm_text)

    def test_fsm_command_lowers_fu_only_symbolic_bind_to_dynamic_controller(self) -> None:
        bind = """design dyn
stage bind
schedule kind=hierarchical
resources {
  fu fu_add0 : FU_ADD
}

region proc_dyn kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = call child=callee class=CTRL ii=symb_ii_v1 delay=symb_delay_v1 start=0 end=symb_delay_v1 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done
  node v2 = add x, y : i32 class=FU_ADD ii=1 delay=1 start=symb_delay_v1 end=symb_delay_v1 + 1 - 1 bind=fu_add0
  node v3 = ret v2 class=CTRL ii=0 delay=0 start=symb_delay_v1 + 1 end=symb_delay_v1 + 1
  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 + 1 end=symb_delay_v1 + 1
  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
  edge data v3 -> v4
  edge seq v1 -> callee hierarchy=true
  edge seq callee -> v1 hierarchy=true
}

region callee kind=procedure parent=proc_dyn {
  node c0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node c1 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
  edge data c0 -> c1
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bind_path = root / "dyn.bind.uhir"
            fsm_path = root / "dyn.fsm.uhir"
            bind_path.write_text(bind, encoding="utf-8")

            self.assertEqual(main(["fsm", str(bind_path), "--encoding", "binary", "-o", str(fsm_path)]), 0)

            fsm_text = fsm_path.read_text(encoding="utf-8")
            self.assertIn("stage fsm", fsm_text)
            self.assertIn("state WAIT_v1 code=2", fsm_text)
            self.assertIn("transition IDLE -> P0 when=req_valid && req_ready && symb_ready_v1", fsm_text)
            self.assertIn("transition WAIT_v1 -> P1 when=symb_done_v1", fsm_text)
            self.assertIn("emit P0 activate=[v1]", fsm_text)
            self.assertIn("emit P1 issue=[fu_add0<-v2]", fsm_text)

    def test_fsm_command_adds_recursive_loop_child_controller(self) -> None:
        bind = """design dyn_loop
stage bind
schedule kind=hierarchical
resources {
  fu fu_hdr0 : FU_HDR
  fu fu_body0 : FU_BODY
  reg r_acc : i32
}

region proc_dyn_loop kind=procedure {
  region_ref loop_hdr
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = loop child=loop_hdr class=CTRL ii=symb_ii_v1 delay=symb_delay_v1 start=0 end=symb_delay_v1 - 1 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done continue_condition=c iterate_when=c exit_when=!c
  node v2 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 end=symb_delay_v1
  edge data v0 -> v1
  edge data v1 -> v2
  edge seq v1 -> loop_hdr hierarchy=true
  edge seq loop_hdr -> v1 hierarchy=true
}

region loop_hdr kind=loop parent=proc_dyn_loop {
  region_ref loop_body
  region_ref loop_exit
  node h0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node h1 = add i, 1:i32 : i32 class=FU_HDR ii=1 delay=1 start=0 end=0 bind=fu_hdr0
  node h2 = branch c true_child=loop_body false_child=loop_exit class=CTRL ii=0 delay=2 timing=static start=1 end=2
  node h3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
  edge data h0 -> h1
  edge data h1 -> h2
  edge data h2 -> h3
  steps [0:2]
  latency 3
}

region loop_body kind=body parent=loop_hdr {
  node b0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
  node b1 = mul x, y : i32 class=FU_BODY ii=1 delay=1 start=1 end=1 bind=fu_body0
  node b2 = add b1, z : i32 class=FU_BODY ii=1 delay=1 start=2 end=2 bind=fu_body0
  node b3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2
  edge data b0 -> b1
  edge data b1 -> b2
  edge data b2 -> b3
  steps [1:2]
  latency 2
  value b1 -> r_acc live=[2:2]
  mux mx_body : input=[r_acc, r_next] output=r_acc sel=body_state
}

region loop_exit kind=empty parent=loop_hdr {
  node e0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
  node e1 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
  edge data e0 -> e1
  steps [1:1]
  latency 0
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bind_path = root / "dyn_loop.bind.uhir"
            fsm_path = root / "dyn_loop.fsm.uhir"
            bind_path.write_text(bind, encoding="utf-8")

            self.assertEqual(main(["fsm", str(bind_path), "--encoding", "binary", "-o", str(fsm_path)]), 0)

            fsm_text = fsm_path.read_text(encoding="utf-8")
            self.assertIn("controller C_loop_hdr encoding=binary protocol=act_done completion_order=in_order overlap=true region=loop_hdr parent_node=v1 {", fsm_text)
            self.assertIn("link C_loop_hdr via=v1 act=[activate, act_valid] ready=[ready, act_ready] done=[completion, done_valid] done_ready=[resp_ready, done_ready]", fsm_text)
            self.assertIn("input  act_valid : i1", fsm_text)
            self.assertIn("output done_valid : i1", fsm_text)
            self.assertIn("transition IDLE -> T0 when=act_valid && act_ready", fsm_text)
            self.assertIn("transition T0 -> T1 when=c", fsm_text)
            self.assertIn("transition T0 -> DONE when=!c", fsm_text)
            self.assertIn("emit IDLE act_ready=true", fsm_text)
            self.assertIn("emit DONE done_valid=true", fsm_text)
            self.assertIn("emit T2 issue=[fu_body0<-b2] latch=[r_acc] select=[mx_body<-body_state]", fsm_text)

    def test_fsm_command_adds_recursive_call_child_controller(self) -> None:
        bind = """design dyn_call
stage bind
schedule kind=hierarchical
resources {
  fu fu_add0 : FU_ADD
  reg r_acc : i32
}

region proc_dyn_call kind=procedure {
  region_ref callee
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = call child=callee class=CTRL ii=symb_ii_v1 delay=symb_delay_v1 start=0 end=symb_delay_v1 timing=symbolic completion=symb_done_v1 ready=symb_ready_v1 handshake=ready_done
  node v2 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 end=symb_delay_v1
  edge data v0 -> v1
  edge data v1 -> v2
  edge seq v1 -> callee hierarchy=true
  edge seq callee -> v1 hierarchy=true
}

region callee kind=procedure parent=proc_dyn_call {
  node c0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node c1 = add x, y : i32 class=FU_ADD ii=1 delay=1 start=0 end=0 bind=fu_add0
  node c2 = ret c1 class=CTRL ii=0 delay=0 start=1 end=1
  node c3 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
  edge data c0 -> c1
  edge data c1 -> c2
  edge data c2 -> c3
  steps [0:1]
  latency 2
  value c1 -> r_acc live=[1:1]
  mux mx_call : input=[r_acc, r_next] output=r_acc sel=call_state
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bind_path = root / "dyn_call.bind.uhir"
            fsm_path = root / "dyn_call.fsm.uhir"
            bind_path.write_text(bind, encoding="utf-8")

            self.assertEqual(main(["fsm", str(bind_path), "--encoding", "binary", "-o", str(fsm_path)]), 0)

            fsm_text = fsm_path.read_text(encoding="utf-8")
            self.assertIn("controller C_callee encoding=binary protocol=act_done completion_order=in_order overlap=true region=callee parent_node=v1 {", fsm_text)
            self.assertIn("link C_callee via=v1 act=[activate, act_valid] ready=[ready, act_ready] done=[completion, done_valid] done_ready=[resp_ready, done_ready]", fsm_text)
            self.assertIn("transition IDLE -> T0 when=act_valid && act_ready", fsm_text)
            self.assertIn("transition T0 -> T1", fsm_text)
            self.assertIn("transition T1 -> DONE", fsm_text)
            self.assertIn("emit T0 issue=[fu_add0<-c1]", fsm_text)
            self.assertIn("emit T1 latch=[r_acc] select=[mx_call<-call_state]", fsm_text)

    def test_fsm_command_adds_recursive_branch_child_controller(self) -> None:
        bind = """design dyn_branch
stage bind
schedule kind=hierarchical
resources {
  fu fu_true0 : FU_TRUE
  fu fu_false0 : FU_FALSE
}

region proc_dyn_branch kind=procedure {
  region_ref bb_true
  region_ref bb_false
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = branch c true_child=bb_true false_child=bb_false class=CTRL ii=0 delay=symb_delay_v1 start=0 end=symb_delay_v1 timing=symbolic completion=symb_done_v1 branch_condition=c
  node v2 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 end=symb_delay_v1
  edge data v0 -> v1
  edge data v1 -> v2
  edge seq v1 -> bb_true hierarchy=true when=true
  edge seq v1 -> bb_false hierarchy=true when=false
  edge seq bb_true -> v1 hierarchy=true when=true
  edge seq bb_false -> v1 hierarchy=true when=false
}

region bb_true kind=basic_block parent=proc_dyn_branch {
  node t0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node t1 = add x, y : i32 class=FU_TRUE ii=1 delay=1 start=0 end=0 bind=fu_true0
  node t2 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1
  edge data t0 -> t1
  edge data t1 -> t2
  steps [0:1]
  latency 2
}

region bb_false kind=basic_block parent=proc_dyn_branch {
  node f0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node f1 = mul a, b : i32 class=FU_FALSE ii=1 delay=1 start=0 end=0 bind=fu_false0
  node f2 = nop role=sink class=CTRL ii=0 delay=0 start=0 end=0
  edge data f0 -> f1
  edge data f1 -> f2
  steps [0:0]
  latency 1
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bind_path = root / "dyn_branch.bind.uhir"
            fsm_path = root / "dyn_branch.fsm.uhir"
            bind_path.write_text(bind, encoding="utf-8")

            self.assertEqual(main(["fsm", str(bind_path), "--encoding", "binary", "-o", str(fsm_path)]), 0)

            fsm_text = fsm_path.read_text(encoding="utf-8")
            self.assertIn("controller C_v1 encoding=binary protocol=act_done completion_order=in_order overlap=true region=bb_true false_region=bb_false parent_node=v1 branch_condition=c {", fsm_text)
            self.assertIn("link C_v1 via=v1 act=[activate, act_valid] ready=[ready, act_ready] done=[completion, done_valid] done_ready=[resp_ready, done_ready]", fsm_text)
            self.assertIn("transition IDLE -> TRUE_T0 when=act_valid && act_ready && c", fsm_text)
            self.assertIn("transition IDLE -> FALSE_T0 when=act_valid && act_ready && !c", fsm_text)
            self.assertIn("emit TRUE_T0 issue=[fu_true0<-t1]", fsm_text)
            self.assertIn("emit FALSE_T0 issue=[fu_false0<-f1]", fsm_text)

    def test_bind_then_fsm_preserves_dynamic_branch_register_latches(self) -> None:
        sched = """design dyn_branch_e2e
stage sched
schedule kind=hierarchical

region proc kind=procedure {
  region_ref bb_true
  region_ref bb_false
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = branch c true_child=bb_true false_child=bb_false class=CTRL ii=0 delay=symb_delay_v1 start=0 end=symb_delay_v1 timing=symbolic completion=symb_done_v1 branch_condition=c
  node v2 = nop role=sink class=CTRL ii=0 delay=0 start=symb_delay_v1 end=symb_delay_v1
  edge data v0 -> v1
  edge data v1 -> v2
  edge seq v1 -> bb_true hierarchy=true when=true
  edge seq v1 -> bb_false hierarchy=true when=false
  edge seq bb_true -> v1 hierarchy=true when=true
  edge seq bb_false -> v1 hierarchy=true when=false
}

region bb_true kind=basic parent=proc {
  node t0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node t1 = add a, b : i32 class=EWMS ii=1 delay=1 start=1 end=1
  node t2 = add t1, k : i32 class=EWMS ii=1 delay=1 start=2 end=2
  node t3 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3
  edge data t0 -> t1
  edge data t1 -> t2
  edge data t2 -> t3
  steps [0:3]
  latency 4
}

region bb_false kind=basic parent=proc {
  node f0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node f1 = add c, d : i32 class=EWMS ii=1 delay=1 start=1 end=1
  node f2 = add f1, m : i32 class=EWMS ii=1 delay=1 start=2 end=2
  node f3 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3
  edge data f0 -> f1
  edge data f1 -> f2
  edge data f2 -> f3
  steps [0:3]
  latency 4
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sched_path = root / "dyn_branch_e2e.sched.uhir"
            bind_path = root / "dyn_branch_e2e.bind.uhir"
            fsm_path = root / "dyn_branch_e2e.fsm.uhir"
            sched_path.write_text(sched, encoding="utf-8")

            self.assertEqual(main(["bind", str(sched_path), "--algo", "compat", "-o", str(bind_path)]), 0)
            self.assertEqual(main(["fsm", str(bind_path), "--encoding", "binary", "-o", str(fsm_path)]), 0)

            bind_text = bind_path.read_text(encoding="utf-8")
            self.assertIn("value t1 -> r_i32_0 live=[2:2]", bind_text)
            self.assertIn("value f1 -> r_i32_0 live=[2:2]", bind_text)

            fsm_text = fsm_path.read_text(encoding="utf-8")
            self.assertIn("emit TRUE_T2 issue=[ewms0<-t2] latch=[r_i32_0]", fsm_text)
            self.assertIn("emit FALSE_T2 issue=[ewms0<-f2] latch=[r_i32_0]", fsm_text)

    def test_uglir_command_lowers_static_fsm(self) -> None:
        fsm = """design add1
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
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fsm_path = root / "add1.fsm.uhir"
            uglir_path = root / "add1.uglir"
            fsm_path.write_text(fsm, encoding="utf-8")

            self.assertEqual(main(["uglir", str(fsm_path), "-o", str(uglir_path)]), 0)

            uglir_text = uglir_path.read_text(encoding="utf-8")
            self.assertIn("stage uglir", uglir_text)
            self.assertIn("input  clk : clock", uglir_text)
            self.assertIn("input  req_valid : i1", uglir_text)
            self.assertIn("output req_ready : i1", uglir_text)
            self.assertIn("inst ewms0 : EWMS", uglir_text)
            self.assertIn("assign req_ready = state_q == 1", uglir_text)
            self.assertIn("ewms0.go(ewms0_go_n)", uglir_text)
            self.assertIn("mux mx_r_i32_0_n : i32 sel=sel_r_i32_0_n {", uglir_text)
            self.assertIn("seq clk {", uglir_text)

    def test_uglir_command_accepts_component_library_json(self) -> None:
        fsm = """design add1
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
        component_library = {
            "components": {
                "ALU": {
                    "kind": "combinational",
                    "ports": {
                        "a": {"dir": "input", "type": "i32"},
                        "b": {"dir": "input", "type": "i32"},
                        "op": {"dir": "input", "type": "u5"},
                        "y": {"dir": "output", "type": "i32"},
                    },
                    "supports": {
                        "add": {
                            "ii": 1,
                            "d": 1,
                            "opcode": 0,
                            "bind": {"a": "operand0", "b": "operand1", "y": "result"},
                        },
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fsm_path = root / "add1.fsm.uhir"
            library_path = root / "ressources.json"
            uglir_path = root / "add1.uglir"
            fsm_path.write_text(fsm, encoding="utf-8")
            library_path.write_text(json.dumps(component_library), encoding="utf-8")

            self.assertEqual(main(["uglir", str(fsm_path), "--ressources", str(library_path), "-o", str(uglir_path)]), 0)

            uglir_text = uglir_path.read_text(encoding="utf-8")
            self.assertIn("ewms0.a(ewms0_a_n)", uglir_text)
            self.assertIn("assign ewms0_a_n = mx_ewms0_a_n", uglir_text)
            self.assertIn("assign ewms0_b_n = mx_ewms0_b_n", uglir_text)
            self.assertIn("mux mx_ewms0_a_n : i32 sel=sel_ewms0_a_n {", uglir_text)
            self.assertIn("ewms0.op(ewms0_op_n)", uglir_text)
            self.assertIn("assign ewms0_op_n = 0", uglir_text)
            self.assertNotIn("ewms0.go(", uglir_text)

    def test_rtl_command_lowers_uglir_to_verilog(self) -> None:
        uglir = """design add1
stage uglir
input  clk : clock
input  rst : i1
input  req_valid : i1
input  resp_ready : i1
input  x : i32
output req_ready : i1
output resp_valid : i1
output result : i32
resources {
  reg state : u5
  net next_state : u5
  net req_fire : i1
  net resp_fire : i1
  inst ewms0 : EWMS
  net ewms0_go : i1
  net ewms0_y : i32
  reg r_i32_0 : i32
  net latch_r_i32_0 : i1
  net sel_r_i32_0 : ctrl
  mux mx_r_i32_0 : i32
}

assign req_fire = req_valid & req_ready

assign resp_fire = resp_valid & resp_ready

assign req_ready = state == 1

assign resp_valid = state == 16

assign next_state = (state == 1 && req_fire) ? 2 : state == 2 ? 4 : state == 4 ? 8 : state == 8 ? 16 : (state == 16 && resp_fire) ? 1 : 1

assign ewms0_go = state == 2

assign latch_r_i32_0 = state == 4

assign sel_r_i32_0 = (state == 4) ? ewms0_y : hold

assign result = r_i32_0

ewms0.go(ewms0_go)

ewms0.y(ewms0_y)

mux mx_r_i32_0 : i32 sel=sel_r_i32_0 {
  hold -> r_i32_0
  ewms0_y -> ewms0_y
}

seq clk {
  if rst {
    state <= 1
  } else {
    state <= next_state
    if latch_r_i32_0 {
      r_i32_0 <= mx_r_i32_0
    }
  }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uglir_path = root / "add1.uglir"
            rtl_path = root / "add1.v"
            uglir_path.write_text(uglir, encoding="utf-8")

            self.assertEqual(main(["rtl", str(uglir_path), "--hdl", "verilog", "-o", str(rtl_path)]), 0)

            verilog_text = rtl_path.read_text(encoding="utf-8")
            self.assertIn("module add1 (", verilog_text)
            self.assertIn("assign req_ready = state == 1;", verilog_text)
            self.assertIn("EWMS ewms0 (", verilog_text)
            self.assertIn("always @(posedge clk) begin", verilog_text)
            self.assertIn("endmodule", verilog_text)

    def test_rtl_command_accepts_wishbone_slave_wrapper(self) -> None:
        uglir = """design add1
stage uglir
input  clk : clock
input  rst : i1
input  req_valid : i1
input  resp_ready : i1
input  x : i32
output req_ready : i1
output resp_valid : i1
output result : i32
resources {
  reg state : u1
  net next_state : u1
}

assign req_ready = 1:i1

assign resp_valid = 1:i1

assign next_state = 0:u1

assign result = x

seq clk {
  if rst {
    state <= 0:u1
  } else {
    state <= next_state
  }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uglir_path = root / "add1.uglir"
            rtl_path = root / "add1_wrap.v"
            uglir_path.write_text(uglir, encoding="utf-8")

            self.assertEqual(
                main(
                    [
                        "rtl",
                        str(uglir_path),
                        "--hdl",
                        "verilog",
                        "--wrap",
                        "slave",
                        "--protocol",
                        "wishbone",
                        "-o",
                        str(rtl_path),
                    ]
                ),
                0,
            )

            verilog_text = rtl_path.read_text(encoding="utf-8")
            self.assertIn("module add1_core (", verilog_text)
            self.assertIn("module add1 #(", verilog_text)
            self.assertIn("input [31:0] wb_adr_i,", verilog_text)
            self.assertIn("parameter [31:0] WB_BASE_ADDR = 32'h0000_0000", verilog_text)
            self.assertIn("localparam [31:0] WB_REG_CONTROL_STATUS = WB_BASE_ADDR + 32'h0000_0000;", verilog_text)
            self.assertIn("assign wb_ack_o = wb_req_n;", verilog_text)
            self.assertIn("assign core_req_valid_n = start_pending_q;", verilog_text)

    def test_rtl_command_accepts_none_memory_wrapper_pair(self) -> None:
        uglir = """design add1
stage uglir
input  clk : clock
input  rst : i1
input  req_valid : i1
input  resp_ready : i1
input  x : i32
output req_ready : i1
output resp_valid : i1
output result : i32
resources {
  reg state : u1
  net next_state : u1
}

assign req_ready = 1:i1

assign resp_valid = 1:i1

assign next_state = 0:u1

assign result = x

seq clk {
  if rst {
    state <= 0:u1
  } else {
    state <= next_state
  }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uglir_path = root / "add1.uglir"
            rtl_path = root / "add1_none.v"
            uglir_path.write_text(uglir, encoding="utf-8")

            self.assertEqual(
                main(
                    [
                        "rtl",
                        str(uglir_path),
                        "--hdl",
                        "verilog",
                        "--wrap",
                        "none",
                        "--protocol",
                        "memory",
                        "-o",
                        str(rtl_path),
                    ]
                ),
                0,
            )

            verilog_text = rtl_path.read_text(encoding="utf-8")
            self.assertIn("module add1 (", verilog_text)
            self.assertNotIn("module add1_core (", verilog_text)
            self.assertIn("input req_valid,", verilog_text)
            self.assertIn("output resp_valid,", verilog_text)

    def test_bind_command_can_render_conflict_dot(self) -> None:
        sched = """design add_pair
stage sched
schedule kind=control_steps

region proc_add_pair kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
  node v2 = add c, d : i32 class=EWMS ii=1 delay=1 start=0 end=0
  node v3 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

  edge data v0 -> v1
  edge data v0 -> v2
  edge data v1 -> v3
  edge data v2 -> v3
  edge data v3 -> v4
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sched_path = root / "add_pair.sched.uhir"
            sched_path.write_text(sched, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["bind", str(sched_path), "--algo", "left_edge", "--dump", "conflict", "--dot"]), 0)

            rendered = stdout.getvalue()
            self.assertIn('digraph "add_pair.bind.dump"', rendered)
            self.assertIn('subgraph "cluster_conflict_operation_ewms"', rendered)
            self.assertIn('label="EWMS operation conflict graph"', rendered)
            self.assertIn('subgraph "cluster_conflict_register_i32"', rendered)
            self.assertIn('"v1" -> "v2" [label="proc_add_pair"', rendered)
            self.assertIn('"v1" [label="v1 add ewms0"', rendered)
            self.assertIn('"reg_v1" [label="reg_v1 reg r_i32_0"', rendered)

    def test_bind_command_supports_compact_dot_labels(self) -> None:
        sched = """design add1
stage sched
schedule kind=control_steps

region proc_add1 kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
  node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sched_path = root / "add1.sched.uhir"
            sched_path.write_text(sched, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["bind", str(sched_path), "--algo", "left_edge", "--dump", "conflict", "--dot", "--compact"]), 0)

            rendered = stdout.getvalue()
            self.assertIn('"v1" [label="v1 + ewms0"', rendered)

    def test_bind_command_requires_algo_for_sched_dump(self) -> None:
        sched = """design add1
stage sched
schedule kind=control_steps

region proc_add1 kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0
  node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sched_path = root / "add1.sched.uhir"
            sched_path.write_text(sched, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["bind", str(sched_path), "--dump", "trp"]), 1)
            self.assertIn("requires --algo", stderr.getvalue())

    def test_bind_command_can_dump_from_bind_input_without_algo(self) -> None:
        bind = """design add1
stage bind
schedule kind=control_steps

resources {
  fu ewms0 : EWMS
  reg r_i32_0 : i32
}

region proc_add1 kind=procedure {
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=ewms0
  node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3

  value v1 -> r_i32_0 live=[1:1]
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bind_path = root / "add1.bind.uhir"
            bind_path.write_text(bind, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["bind", str(bind_path), "--dump", "trp"]), 0)
            rendered = stdout.getvalue()
            self.assertIn("bind_dump trp", rendered)
            self.assertIn("region proc_add1 (time-resource plane)", rendered)
            self.assertIn("r_i32_0", rendered)

    def test_bind_command_can_dump_dfgsb_unroll(self) -> None:
        bind = """design add1
stage bind
schedule kind=control_steps

resources {
  fu ewms0 : EWMS
  reg r_i32_0 : i32
}

region proc_add1 kind=procedure {
  map v1 <- t1_0
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = add a, b : i32 class=EWMS ii=1 delay=1 start=0 end=0 bind=ewms0
  node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1
  node v3 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3

  value t1_0 -> r_i32_0 live=[1:1]
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bind_path = root / "add1.bind.uhir"
            bind_path.write_text(bind, encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["bind", str(bind_path), "--dump", "dfgsb_unroll"]), 0)
            rendered = stdout.getvalue()
            self.assertIn("bind_dump dfgsb_unroll", rendered)
            self.assertIn("global (dataflow graph with schedule and binding, unrolled)", rendered)
            self.assertIn("reg 1", rendered)

    def test_bind_command_reuses_resources_across_mutually_exclusive_branch_sgus(self) -> None:
        sched = """design branch_share
stage sched
schedule kind=hierarchical

region proc_branch_share kind=procedure {
  region_ref bb_true
  region_ref bb_false
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = lt a, b : i1 class=EWMS ii=1 delay=1 start=0 end=0
  node v2 = branch v1 true_child=bb_true false_child=bb_false class=CTRL ii=0 delay=1 start=1 end=1
  node v3 = ret x class=CTRL ii=0 delay=0 start=3 end=3
  node v4 = nop role=sink class=CTRL ii=0 delay=0 start=4 end=4

  edge data v0 -> v1
  edge data v1 -> v2
  edge data v2 -> v3
  edge data v3 -> v4
}

region bb_true kind=basic parent=proc_branch_share {
  node t0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
  node t1 = add a, b : i32 class=EWMS ii=1 delay=1 start=2 end=2
  node t2 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

  edge data t0 -> t1
  edge data t1 -> t2
}

region bb_false kind=basic parent=proc_branch_share {
  node f0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
  node f1 = add c, d : i32 class=EWMS ii=1 delay=1 start=2 end=2
  node f2 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

  edge data f0 -> f1
  edge data f1 -> f2
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sched_path = root / "branch_share.sched.uhir"
            bind_path = root / "branch_share.bind.uhir"
            sched_path.write_text(sched, encoding="utf-8")

            self.assertEqual(main(["bind", str(sched_path), "--algo", "left_edge", "-o", str(bind_path)]), 0)

            bind_text = bind_path.read_text(encoding="utf-8")
            self.assertIn("fu ewms0 : EWMS", bind_text)
            self.assertNotIn("fu ewms1 : EWMS", bind_text)
            self.assertIn("node t1 = add a, b : i32 class=EWMS ii=1 delay=1 start=2 end=2 bind=ewms0", bind_text)
            self.assertIn("node f1 = add c, d : i32 class=EWMS ii=1 delay=1 start=2 end=2 bind=ewms0", bind_text)

    def test_bind_command_flatten_rejects_non_static_loop_design(self) -> None:
        sched = """design dynamic_loop
stage sched
schedule kind=hierarchical

region proc_dynamic_loop kind=procedure {
  region_ref loop_header_1
  node v0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node v1 = loop child=loop_header_1 class=CTRL ii=0 delay=4 start=0 end=4
  node v2 = nop role=sink class=CTRL ii=0 delay=0 start=5 end=5

  edge data v0 -> v1
  edge data v1 -> v2
}

region loop_header_1 kind=loop parent=proc_dynamic_loop {
  node h0 = nop role=source class=CTRL ii=0 delay=0 start=0 end=0
  node h1 = branch c true_child=loop_body_1 false_child=loop_exit_1 class=CTRL ii=0 delay=1 start=1 end=1
  node h2 = nop role=sink class=CTRL ii=0 delay=0 start=2 end=2

  edge data h0 -> h1
  edge data h1 -> h2
}

region loop_body_1 kind=basic parent=loop_header_1 {
  node b0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
  node b1 = add a, b : i32 class=EWMS ii=1 delay=1 start=2 end=2
  node b2 = nop role=sink class=CTRL ii=0 delay=0 start=3 end=3

  edge data b0 -> b1
  edge data b1 -> b2
}

region loop_exit_1 kind=basic parent=loop_header_1 {
  node e0 = nop role=source class=CTRL ii=0 delay=0 start=1 end=1
  node e1 = nop role=sink class=CTRL ii=0 delay=0 start=1 end=1

  edge data e0 -> e1
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sched_path = root / "dynamic_loop.sched.uhir"
            sched_path.write_text(sched, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["bind", str(sched_path), "--algo", "left_edge", "--flatten"]), 1)
            self.assertIn("fully static design", stderr.getvalue())
