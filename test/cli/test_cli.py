from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from click.testing import CliRunner

from uhls.cli import cli, main


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

    def test_opt_command_exposes_pipeline_but_reports_unimplemented_passes(self) -> None:
        uir = """func add1(x:i32) -> i32

block entry:
    t0:i32 = add x, 1
    ret t0
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "add1.uir"
            out_path = Path(tmpdir) / "add1_opt.uir"
            path.write_text(uir, encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["opt", str(path), "-p", "constprop,dce", "-o", str(out_path)]), 1)
            self.assertIn("implement constant propagation here", stderr.getvalue())

            stderr_stdout = io.StringIO()
            with redirect_stderr(stderr_stdout):
                self.assertEqual(main(["opt", str(path), "-p", "constprop,dce"]), 1)
            self.assertIn("implement constant propagation here", stderr_stdout.getvalue())

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
            self.assertEqual(main(["hls-sched", "foo.uir", "--algo", "list", "-o", "foo.sched"]), 1)
        self.assertIn("not implemented yet", stderr.getvalue())

    def test_opt_help_lists_implemented_and_registered_passes(self) -> None:
        result = CliRunner().invoke(cli, ["opt", "-h"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Implemented passes: simplify_cfg, inline_calls, copyprop, cse, dce,", result.output)
        self.assertIn("prune_functions.", result.output)
        self.assertIn("Registered pass", result.output)
        self.assertIn("simplify_cfg, inline_calls, constprop,", result.output)
        self.assertIn("copyprop, cse, dce, prune_functions.", result.output)
        self.assertIn("constprop, copyprop, cse,", result.output)
        self.assertIn("dce, prune_functions.", result.output)
        self.assertIn("External pass", result.output)
        self.assertIn("/path/to/pass.py:Symbol", result.output)
        self.assertIn("Example: uhls opt input.uir -p", result.output)
        self.assertIn("simplify_cfg,inline_calls --pass-arg dot4 -o", result.output)
        self.assertIn("output.uir", result.output)
        self.assertIn("--pass-arg TEXT", result.output)

    def test_opt_help_for_one_pass_prints_brief_description(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["opt", "-h", "simplify_cfg"]), 0)

        output = stdout.getvalue()
        self.assertIn("simplify_cfg:", output)
        self.assertIn("pruning unreachable blocks", output)
        self.assertIn("status: implemented", output)
        self.assertIn("example: uhls opt input.uir -p simplify_cfg -o output.uir", output)

    def test_opt_help_for_inline_calls_mentions_main_boundary(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["opt", "-h", "inline_calls"]), 0)

        output = stdout.getvalue()
        self.assertIn("Calls made directly from 'main' are left intact.", output)
        self.assertIn("Repeat --pass-arg with callee names", output)

    def test_run_help_includes_example_invoke(self) -> None:
        result = CliRunner().invoke(cli, ["run", "-h"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Execute IR with the interpreter.", result.output)
        self.assertIn("Examples:", result.output)
        self.assertIn("uhls run input.uir --function add1 --arg x=7", result.output)
        self.assertIn("--arg A=[1,2,3,4] --arg B=[4,3,2,1]", result.output)
        self.assertIn("Function input. Scalar: name=7. Array: name=[1,2,3,4]:i32.", result.output)
