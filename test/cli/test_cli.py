from __future__ import annotations

import io
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
                self.assertEqual(main(["verify", str(uir_path)]), 0)
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

            self.assertEqual(main(["sched", str(alloc_path), "-o", str(sched_path)]), 0)

            sched_text = sched_path.read_text(encoding="utf-8")
            self.assertIn("stage sched", sched_text)
            self.assertIn("schedule kind=hierarchical", sched_text)
            self.assertIn("node v1 = add x, 1 : i32 class=FU_FAST_ADD ii=1 delay=1 start=0 end=0", sched_text)
            self.assertIn("node v2 = ret v1 class=CTRL ii=0 delay=0 start=1 end=1", sched_text)
            self.assertIn("steps 0..1", sched_text)
            self.assertIn("latency 2", sched_text)

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
            self.assertIn("steps 4..6", sched_text)
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

    def test_verify_accepts_local_arrays_and_print(self) -> None:
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
                self.assertEqual(main(["verify", str(path)]), 0)
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

    def test_hls_placeholder_commands_exist(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(main(["hls-bind", "foo.sched.uhir", "-o", "foo.bind.uhir"]), 1)
        self.assertIn("'hls-bind' is not implemented yet", stderr.getvalue())
