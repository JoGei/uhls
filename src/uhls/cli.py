"""Command-line interface for the µhLS toolkit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from inspect import Parameter as InspectParameter
from inspect import signature
from pathlib import Path
from typing import Callable

import click

from uhls.backend.hls import (
    BIND_DUMP_KINDS,
    CallableSGUScheduler,
    FSM_ENCODINGS,
    bind_dump_to_dot,
    binding_to_dot,
    builtin_binder_names,
    builtin_scheduler_names,
    create_builtin_binder,
    create_builtin_scheduler,
    fsm_to_dot,
    format_bind_dump,
    lower_bind_to_fsm,
    lower_fsm_to_uglir,
    parse_bind_dump_spec,
)
from uhls.backend.uhir import (
    ExecutabilityGraph,
    GOptPassSpec,
    UHIRParseError,
    builtin_gopt_pass_names,
    builtin_gopt_specs,
    create_builtin_gopt_pass,
    dummy_executability_graph,
    executability_graph_from_uhir,
    format_uhir,
    lower_alloc_to_sched,
    lower_seq_to_alloc,
    lower_sched_to_bind,
    lower_module_to_seq,
    parse_uhir_file,
    run_gopt_passes,
    to_dot as to_uhir_dot,
)
from uhls.backend.hls.alloc import executability_graph_to_dot, format_executability_graph
from uhls.frontend import lower_source_to_uir
from uhls.interpreter import InterpreterError, run_uir
from uhls.middleend.uir import ArrayType, IRParseError, format_module, normalize_type, parse_module, verify_module
from uhls.middleend.passes.analyze import build_cfg, build_dfg, control_flow
from uhls.middleend.passes.opt import (
    CSEPass,
    ConstPropPass,
    CopyPropPass,
    DCEPass,
    InlineCallsPass,
    PruneFunctionsPass,
    SimplifyCFGPass,
)
from uhls.middleend.passes.util import PassContext, PassManager
from uhls.middleend.passes.util.dot import (
    to_basic_block_dfg_dot,
    to_cdfg_dot,
    to_dfg_dot,
    to_dot,
    to_module_cdfg_dot,
    to_module_dfg_dot,
)


class CLIError(RuntimeError):
    """Raised for user-facing command-line failures."""


@dataclass(frozen=True)
class _NamedPassAdapter:
    """Adapter for external pass objects that provide ``run`` but no ``name``."""

    name: str
    delegate: object

    def run(self, ir: object, context: PassContext) -> object:
        return _invoke_external_pass_run(self.delegate, ir, context)


@dataclass(frozen=True)
class OptPassSpec:
    """Metadata for one pass exposed through ``uhls opt``."""

    name: str
    factory: Callable[[], object]
    description: str
    example: str
    aliases: tuple[str, ...] = ()


_OPT_PASS_SPECS: tuple[OptPassSpec, ...] = (
    OptPassSpec(
        name="simplify_cfg",
        factory=SimplifyCFGPass,
        description="Clean up CFG structure by pruning unreachable blocks, collapsing trivial jumps, and normalizing block order.",
        example="uhls opt input.uir -p simplify_cfg -o output.uir",
        aliases=("simplify",),
    ),
    OptPassSpec(
        name="inline_calls",
        factory=InlineCallsPass,
        description=(
            "Inline direct module-local calls into their callers and then simplify the resulting CFG. "
            "Calls made directly from 'main' are left intact. "
            "Repeat --pass-arg with callee names to restrict which functions are inlined."
        ),
        example="uhls opt input.uir -p inline_calls -o output.uir",
        aliases=("inline",),
    ),
    OptPassSpec(
        name="constprop",
        factory=ConstPropPass,
        description="Propagate constant values through the IR to simplify computations and expose dead code.",
        example="uhls opt input.uir -p constprop -o output.uir",
        aliases=("const_prop",),
    ),
    OptPassSpec(
        name="copyprop",
        factory=CopyPropPass,
        description="Replace copied temporaries with their original values when it is safe to do so.",
        example="uhls opt input.uir -p copyprop -o output.uir",
        aliases=("copy_prop",),
    ),
    OptPassSpec(
        name="cse",
        factory=CSEPass,
        description="Eliminate repeated equivalent computations by reusing previously computed values.",
        example="uhls opt input.uir -p cse -o output.uir",
    ),
    OptPassSpec(
        name="dce",
        factory=DCEPass,
        description="Remove operations whose results are unused and that have no required side effects.",
        example="uhls opt input.uir -p dce -o output.uir",
    ),
    OptPassSpec(
        name="prune_functions",
        factory=PruneFunctionsPass,
        description="Remove functions that are unreachable from module entry roots such as 'main'.",
        example="uhls opt input.uir -p prune_functions -o output.uir",
        aliases=("prune",),
    ),
)


def _implemented_opt_help() -> str:
    implemented = _implemented_opt_pass_names()
    return ", ".join(implemented) if implemented else "none"


def _registered_opt_help() -> str:
    return ", ".join(spec.name for spec in _OPT_PASS_SPECS) if _OPT_PASS_SPECS else "none"


def _implemented_opt_pass_names() -> list[str]:
    return [spec.name for spec in _OPT_PASS_SPECS if _is_implemented_opt_pass(spec)]


def _callable_is_placeholder(callback: object) -> bool:
    if callback is None:
        return False
    code = getattr(callback, "__code__", None)
    if code is None:
        return False
    names = set(code.co_names)
    constants = set(item for item in code.co_consts if isinstance(item, str))
    return "NotImplementedError" in names or any("implement " in item for item in constants)


def _is_implemented_opt_pass(spec: OptPassSpec) -> bool:
    try:
        pass_like = spec.factory()
    except NotImplementedError:
        return False

    run = getattr(pass_like, "run", None)
    if run is None:
        return True

    transform = getattr(pass_like, "transform", None)
    function_transform = getattr(pass_like, "function_transform", None)
    module_transform = getattr(pass_like, "module_transform", None)
    return not any(
        _callable_is_placeholder(callback)
        for callback in (transform, function_transform, module_transform)
    )


def _opt_specs_by_name() -> dict[str, OptPassSpec]:
    return {spec.name: spec for spec in _OPT_PASS_SPECS}


def _opt_specs_by_cli_name() -> dict[str, OptPassSpec]:
    by_name: dict[str, OptPassSpec] = {}
    for spec in _OPT_PASS_SPECS:
        by_name[spec.name] = spec
        for alias in spec.aliases:
            by_name[alias] = spec
    return by_name


def _opt_pass_canonical_name(name: str) -> str | None:
    spec = _opt_specs_by_cli_name().get(name)
    return None if spec is None else spec.name


def _implemented_gopt_help() -> str:
    implemented = _implemented_gopt_pass_names()
    return ", ".join(implemented) if implemented else "none"


def _registered_gopt_help() -> str:
    specs = builtin_gopt_specs()
    return ", ".join(spec.name for spec in specs) if specs else "none"


def _implemented_gopt_pass_names() -> list[str]:
    return [spec.name for spec in builtin_gopt_specs() if _is_implemented_gopt_pass(spec)]


def _is_implemented_gopt_pass(spec: GOptPassSpec) -> bool:
    try:
        pass_like = spec.factory()
    except NotImplementedError:
        return False

    run = getattr(pass_like, "run", None)
    if run is None:
        return True
    return not _callable_is_placeholder(run)


def _gopt_specs_by_name() -> dict[str, GOptPassSpec]:
    return {spec.name: spec for spec in builtin_gopt_specs()}


def _gopt_specs_by_cli_name() -> dict[str, GOptPassSpec]:
    by_name: dict[str, GOptPassSpec] = {}
    for spec in builtin_gopt_specs():
        by_name[spec.name] = spec
        for alias in spec.aliases:
            by_name[alias] = spec
    return by_name


def _gopt_pass_canonical_name(name: str) -> str | None:
    spec = _gopt_specs_by_cli_name().get(name)
    return None if spec is None else spec.name


def _maybe_handle_opt_pass_help(argv: list[str]) -> int | None:
    if len(argv) != 3 or argv[0] != "opt" or argv[1] not in {"-h", "--help"}:
        return None

    canonical = _opt_pass_canonical_name(argv[2].strip().lower().replace("-", "_"))
    if canonical is None:
        click.echo(f"error: unknown optimization pass '{argv[2]}'", err=True)
        return 1

    spec = _opt_specs_by_name()[canonical]
    implemented = _is_implemented_opt_pass(spec)
    click.echo(f"{canonical}: {spec.description}")
    click.echo(f"status: {'implemented' if implemented else 'registered but not implemented yet'}")
    click.echo(f"example: {spec.example}")

    if spec.aliases:
        click.echo(f"aliases: {', '.join(sorted(spec.aliases))}")
    return 0


def _maybe_handle_gopt_pass_help(argv: list[str]) -> int | None:
    if len(argv) != 3 or argv[0] != "gopt" or argv[1] not in {"-h", "--help"}:
        return None

    canonical = _gopt_pass_canonical_name(argv[2].strip().lower().replace("-", "_"))
    if canonical is None:
        click.echo(f"error: unknown graph optimization pass '{argv[2]}'", err=True)
        return 1

    spec = _gopt_specs_by_name()[canonical]
    implemented = _is_implemented_gopt_pass(spec)
    click.echo(f"{canonical}: {spec.description}")
    click.echo(f"status: {'implemented' if implemented else 'registered but not implemented yet'}")
    click.echo(f"example: {spec.example}")
    if spec.aliases:
        click.echo(f"aliases: {', '.join(sorted(spec.aliases))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the µhLS CLI and return a process exit code."""
    active_argv = list(sys.argv[1:] if argv is None else argv)
    opt_help_exit = _maybe_handle_opt_pass_help(active_argv)
    if opt_help_exit is not None:
        return opt_help_exit
    gopt_help_exit = _maybe_handle_gopt_pass_help(active_argv)
    if gopt_help_exit is not None:
        return gopt_help_exit

    try:
        cli.main(args=active_argv, prog_name="uhls", standalone_mode=False)
        return 0
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return int(exc.exit_code)
    except click.Abort:
        click.echo("Aborted!", err=True)
        return 1
    except KeyError as exc:
        message = exc.args[0] if exc.args else str(exc)
        click.echo(f"error: {message}", err=True)
        return 1
    except (CLIError, IRParseError, UHIRParseError, InterpreterError, NotImplementedError, ValueError) as exc:
        click.echo(f"error: {exc}", err=True)
        return 1


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """µhLS command-line toolkit."""


@cli.command("parse")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def parse_cmd(source: Path, output: Path | None) -> None:
    """Parse µC and lower to canonical µIR."""
    module = lower_source_to_uir(source.read_text(encoding="utf-8"))
    _write_or_print_text(format_module(module), output)


@cli.command("verify")
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def verify_cmd(input_path: Path) -> None:
    """Verify canonical IR."""
    module = _load_ir_file(input_path)
    verify_module(module, require_ssa=True, allow_calls=True)
    click.echo("ok")


@cli.command("cfg")
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--function", "function_name")
@click.option("--dot", "emit_dot", is_flag=True, help="Render Graphviz DOT.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def cfg_cmd(input_path: Path, function_name: str | None, emit_dot: bool, output: Path | None) -> None:
    """Inspect CFG structure."""
    module = _load_ir_file(input_path)
    functions = _selected_functions(module, function_name)
    if emit_dot:
        if function_name is None and len(functions) > 1:
            text = to_dot(module)
        else:
            text = "\n\n".join(to_dot(function) for function in functions)
    else:
        text = "\n\n".join(_format_cfg_summary(function) for function in functions)
    _write_or_print_text(text, output)


@cli.command("dfg")
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--function", "function_name")
@click.option("--block", "block_name", help="Restrict DFG output to one basic block.")
@click.option("--dot", "emit_dot", is_flag=True, help="Render Graphviz DOT.")
@click.option("--compact", is_flag=True, help="Use compact DOT labels for DFG nodes and edges.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def dfg_cmd(
    input_path: Path,
    function_name: str | None,
    block_name: str | None,
    emit_dot: bool,
    compact: bool,
    output: Path | None,
) -> None:
    """Inspect block-local DFG structure."""
    module = _load_ir_file(input_path)
    functions = _selected_functions(module, function_name)
    if emit_dot:
        if function_name is None and block_name is None and len(functions) > 1:
            text = to_module_dfg_dot(module, compact=compact)
        else:
            text = "\n\n".join(_render_dfg_dot(function, block_name, compact=compact) for function in functions)
    else:
        text = "\n\n".join(_format_dfg_summary(function, block_name) for function in functions)
    _write_or_print_text(text, output)


@cli.command("cdfg")
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--function", "function_name")
@click.option("--dot", "emit_dot", is_flag=True, help="Render Graphviz DOT.")
@click.option("--compact", is_flag=True, help="Use compact DOT labels for embedded DFGs.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def cdfg_cmd(
    input_path: Path,
    function_name: str | None,
    emit_dot: bool,
    compact: bool,
    output: Path | None,
) -> None:
    """Inspect combined control/data-flow graph structure."""
    module = _load_ir_file(input_path)
    functions = _selected_functions(module, function_name)
    if emit_dot:
        if function_name is None and len(functions) > 1:
            text = to_module_cdfg_dot(module, compact=compact)
        else:
            text = "\n\n".join(to_cdfg_dot(function, compact=compact) for function in functions)
    else:
        text = "\n\n".join(_format_cdfg_summary(function) for function in functions)
    _write_or_print_text(text, output)


@cli.command("seq")
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--top", "top_name", help="Root function to lower when the input µIR module has multiple functions.")
@click.option("--dot", "emit_dot", is_flag=True, help="Render Graphviz DOT from one .seq.uhir file.")
@click.option("--compact", is_flag=True, help="Use compact DOT labels for sequencing-graph nodes.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def seq_cmd(
    input_path: Path,
    top_name: str | None,
    emit_dot: bool,
    compact: bool,
    output: Path | None,
) -> None:
    """Build or visualize hierarchical HLS sequencing graphs."""
    if emit_dot:
        if input_path.suffix == ".uir":
            module = _load_ir_file(input_path)
            if not hasattr(module, "functions"):
                raise CLIError(f"'seq --dot' expects canonical µIR input, got {type(module).__name__}")
            design = lower_module_to_seq(module, top=top_name)
        else:
            design = parse_uhir_file(input_path)
            if design.stage != "seq":
                raise CLIError(f"'seq --dot' expects a seq-stage µhIR file, got stage '{design.stage}'")
        _write_or_print_text(to_uhir_dot(design, compact=compact), output)
        return

    module = _load_ir_file(input_path)
    if not hasattr(module, "functions"):
        raise CLIError(f"'seq' expects canonical µIR input, got {type(module).__name__}")
    design = lower_module_to_seq(module, top=top_name)
    _write_or_print_text(format_uhir(design), output)


@cli.command(
    "alloc",
    help=(
        "Lower seq-stage µhIR with -exg, inspect an executability graph when no input is given, "
        "or emit a starter graph with --gen_dummy_exg.\n"
        "\n"
        "Examples:\n"
        "\n"
        "\b\n"
        "  uhls alloc input.seq.uhir -exg graph.json\n"
        "\n"
        "\b\n"
        "  uhls alloc input.seq.uhir -exg graph.uhir\n"
        "\n"
        "\b\n"
        "  uhls alloc input.seq.uhir -exg graph.json --dot\n"
        "\n"
        "\b\n"
        "  uhls alloc -exg graph.json\n"
        "\n"
        "\b\n"
        "  uhls alloc -exg graph.uhir --dot\n"
        "\n"
        "\b\n"
        "  uhls alloc --gen_dummy_exg\n"
    ),
)
@click.argument("input_path", metavar="input", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-exg",
    "--executability-graph",
    "executability_graph_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Executability graph in JSON or exg-stage µhIR form, used for alloc lowering or inspected directly when no input file is given.",
)
@click.option(
    "-dummy_exg",
    "--gen_dummy_exg",
    "gen_dummy_exg",
    is_flag=True,
    help="Emit a starter executability graph and exit; with --dot, render that starter graph as DOT.",
)
@click.option(
    "--dot",
    "emit_dot",
    is_flag=True,
    help="Render Graphviz DOT for the alloc result, or for the executability graph when no input file is given.",
)
@click.option(
    "--algo",
    "allocation_algorithm",
    type=click.Choice(["min_delay", "min_ii"], case_sensitive=False),
    default="min_delay",
    show_default=True,
    help="Allocation strategy used when choosing one FU candidate per opcode.",
)
@click.option("--compact", is_flag=True, help="Use compact DOT labels for alloc-region rendering.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def alloc_cmd(
    input_path: Path | None,
    executability_graph_path: Path | None,
    gen_dummy_exg: bool,
    emit_dot: bool,
    allocation_algorithm: str,
    compact: bool,
    output: Path | None,
) -> None:
    """Allocate seq-stage µhIR or inspect executability graphs."""
    if gen_dummy_exg:
        graph = dummy_executability_graph()
        _write_or_print_text(executability_graph_to_dot(graph) if emit_dot else _format_executability_graph_json(graph), output)
        return

    if executability_graph_path is None:
        raise CLIError("'alloc' requires -exg/--executability-graph unless -dummy_exg/--gen_dummy_exg is used")
    if input_path is None:
        graph = _load_executability_graph(executability_graph_path)
        _write_or_print_text(executability_graph_to_dot(graph) if emit_dot else format_executability_graph(graph), output)
        return

    design = parse_uhir_file(input_path)
    if emit_dot:
        if design.stage == "seq":
            design = lower_seq_to_alloc(
                design,
                executability_graph=_load_executability_graph(executability_graph_path),
                algorithm=allocation_algorithm,
            )
        elif design.stage != "alloc":
            raise CLIError(f"'alloc --dot' expects seq/alloc-stage µhIR input, got stage '{design.stage}'")
        _write_or_print_text(to_uhir_dot(design, compact=compact), output)
        return

    if design.stage != "seq":
        raise CLIError(f"'alloc' expects seq-stage µhIR input, got stage '{design.stage}'")
    allocated = lower_seq_to_alloc(
        design,
        executability_graph=_load_executability_graph(executability_graph_path),
        algorithm=allocation_algorithm,
    )
    _write_or_print_text(format_uhir(allocated), output)


@cli.command(
    "opt",
    help=(
        "Run optimization pass pipelines.\n"
        "\n"
        f"Implemented passes: {_implemented_opt_help()}.\n"
        f"Registered passes: {_registered_opt_help()}.\n"
        "External pass syntax: /path/to/pass.py:Symbol\n"
        "\n"
        "Example: uhls opt input.uir -p simplify_cfg,inline_calls --pass-arg dot4 -o output.uir\n"
        "\n"
        "More examples:\n"
        "\n"
        "\b\n"
        "  uhls opt input.uir -p simplify_cfg,constprop -o output.uir\n"
        "\n"
        "\b\n"
        "  uhls opt input.uir -p simplify_cfg,inline_calls --pass-arg dot4 -o output.uir\n"
        "\n"
        "\b\n"
        "  uhls opt input.uir -p /path/to/pass.py:Symbol -o output.uir\n"
    ),
)
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-p",
    "--passes",
    required=True,
    help=(
        "Comma-separated optimization passes. "
        "Use registered pass names or external /path/to/pass.py:Symbol values."
    ),
)
@click.option(
    "--pass-arg",
    "pass_args",
    multiple=True,
    help="Shared pass argument forwarded to every pass in the pipeline. Repeat to pass multiple values.",
)
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def opt_cmd(input_path: Path, passes: str, pass_args: tuple[str, ...], output: Path | None) -> None:
    """Run optimization pass pipelines."""
    module = _load_ir_file(input_path)
    pipeline = [_lookup_opt_pass(name, pass_args) for name in passes.split(",") if name.strip()]
    if not pipeline:
        raise CLIError("no optimization passes were selected")
    context = PassContext(pass_args=tuple(pass_args))
    context.data["pass_args"] = list(pass_args)
    try:
        optimized = PassManager(pipeline).run(module, context)
    except (CLIError, NotImplementedError, ValueError):
        raise
    except Exception as exc:
        raise CLIError(f"optimization pipeline failed: {exc}") from exc
    if not hasattr(optimized, "functions"):
        raise CLIError(
            f"optimization pipeline must return a module-like IR object, got {type(optimized).__name__}"
        )
    for warning in context.data.get("warnings", []):
        click.echo(f"warning: {warning}", err=True)
    _write_or_print_text(format_module(optimized), output)


@cli.command(
    "gopt",
    help=(
        "Run µhIR graph-optimization pass pipelines.\n"
        "\n"
        f"Implemented passes: {_implemented_gopt_help()}.\n"
        f"Registered passes: {_registered_gopt_help()}.\n"
        "External pass syntax: /path/to/pass.py:Symbol\n"
        "\n"
        "Example: uhls gopt input.seq.uhir -p infer_loops,translate_loop_dialect -o output.seq.uhir\n"
        "\n"
        "More examples:\n"
        "\n"
        "\b\n"
        "  uhls gopt input.seq.uhir -p infer_loops,translate_loop_dialect -o output.seq.uhir\n"
        "\n"
        "\b\n"
        "  uhls gopt input.seq.uhir -p infer_loops,translate_loop_dialect,infer_static -o output.seq.uhir\n"
        "\n"
        "\b\n"
        "  uhls gopt input.seq.uhir -p infer_loops,translate_loop_dialect,infer_static,simplify_static_control -o output.seq.uhir\n"
        "\n"
        "\b\n"
        "  uhls gopt input.seq.uhir -p infer_loops,translate_loop_dialect,infer_static,simplify_static_control --dot -o output.dot\n"
        "\n"
        "\b\n"
        "  uhls gopt input.seq.uhir -p /path/to/pass.py:Symbol -o output.seq.uhir\n"
    ),
)
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-p",
    "--passes",
    required=True,
    help=(
        "Comma-separated µhIR graph-optimization passes. "
        "Use registered pass names or external /path/to/pass.py:Symbol values. "
        "Use 'uhls gopt -h <pass>' for pass-specific help."
    ),
)
@click.option(
    "--pass-arg",
    "pass_args",
    multiple=True,
    help="Shared pass argument forwarded to every pass in the pipeline. Repeat to pass multiple values.",
)
@click.option("--dot", "emit_dot", is_flag=True, help="Render Graphviz DOT for the optimized µhIR.")
@click.option("--compact", is_flag=True, help="Use compact DOT labels for µhIR rendering.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def gopt_cmd(
    input_path: Path,
    passes: str,
    pass_args: tuple[str, ...],
    emit_dot: bool,
    compact: bool,
    output: Path | None,
) -> None:
    """Run µhIR graph-optimization pass pipelines."""
    if input_path.suffix == ".dot":
        raise CLIError(
            f"'gopt' expects one µhIR input file, not Graphviz DOT: '{input_path}'. "
            "Pass the underlying .uhir artifact instead."
        )
    try:
        design = parse_uhir_file(input_path)
    except UHIRParseError as exc:
        raise CLIError(f"'gopt' expects one µhIR input file; failed to parse '{input_path}': {exc}") from exc
    pipeline = [_lookup_gopt_pass(name, pass_args) for name in passes.split(",") if name.strip()]
    if not pipeline:
        raise CLIError("no graph optimization passes were selected")
    try:
        optimized = run_gopt_passes(design, pipeline, pass_args=pass_args)
    except (CLIError, NotImplementedError, ValueError):
        raise
    except Exception as exc:
        raise CLIError(f"graph optimization pipeline failed: {exc}") from exc
    _write_or_print_text(to_uhir_dot(optimized, compact=compact) if emit_dot else format_uhir(optimized), output)


@cli.command(
    "run",
    help=(
        "Execute IR with the interpreter.\n"
        "\n"
        "Examples:\n"
        "\n"
        "\b\n"
        "  uhls run input.uir --function add1 --arg x=7\n"
        "\n"
        "\b\n"
        "  uhls run input.uir --function dot4 --arg A=[1,2,3,4] --arg B=[4,3,2,1]\n"
    ),
)
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--function", "function_name")
@click.option(
    "--arg",
    "arguments",
    multiple=True,
    help="Function input. Scalar: name=7. Array: name=[1,2,3,4]:i32.",
)
@click.option("--array", "legacy_arrays", multiple=True, hidden=True)
@click.option("--trace", is_flag=True)
def run_cmd(
    input_path: Path,
    function_name: str | None,
    arguments: tuple[str, ...],
    legacy_arrays: tuple[str, ...],
    trace: bool,
) -> None:
    """Execute IR with the interpreter."""
    module = _load_ir_file(input_path)
    function = _select_function_for_run(module, function_name)
    raw_arguments = _parse_run_argument_items(list(legacy_arrays))
    raw_arguments.update(_parse_run_argument_items(list(arguments)))
    scalar_arguments, array_arguments = _parse_run_arguments(function, raw_arguments)
    _validate_run_invocation(function, scalar_arguments, array_arguments)
    result = run_uir(function, arguments=scalar_arguments, arrays=array_arguments, module=module, trace=trace)

    if trace:
        for event in result.state.trace:
            block = "" if event.block is None else f"[{event.block}] "
            opcode = "" if event.opcode is None else f"{event.opcode} "
            detail = "" if event.detail is None else event.detail
            click.echo(f"{event.step}: {block}{opcode}{detail}".rstrip())
    for line in result.state.stdout:
        click.echo(line)
    if result.return_value is not None:
        click.echo(result.return_value)


@cli.command("sched")
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--algo",
    required=True,
    help="Flat SGU scheduler name or external /path/to/scheduler.py:Symbol.",
)
@click.option(
    "--sgu_latency_max",
    help="External scheduler latency target map: region:value,... or asap[+slack].",
)
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def sched_cmd(input_path: Path, algo: str | None, sgu_latency_max: str | None, output: Path | None) -> None:
    """Schedule alloc-stage µhIR hierarchically."""
    design = parse_uhir_file(input_path)
    if design.stage != "alloc":
        raise CLIError(f"'sched' expects alloc-stage µhIR input, got stage '{design.stage}'")
    scheduler_kwargs = _parse_scheduler_cli_kwargs(sgu_latency_max=sgu_latency_max)
    scheduled = lower_alloc_to_sched(design, scheduler=_lookup_flat_sgu_scheduler(algo, scheduler_kwargs=scheduler_kwargs))
    _write_or_print_text(format_uhir(scheduled), output)


@cli.command(
    "bind",
    help=(
        "Bind sched-stage µhIR operations to concrete resource instances.\n"
        "\n"
        "Built-in binders:\n"
        "  left_edge  concrete interval-based FU+register binding for fully scheduled static designs\n"
        "  compat     hierarchy/control compatibility-based FU-only binding for symbolic schedules\n"
        "\n"
        "Note: bind analysis dumps currently require concrete bind timing.\n"
        "\n"
        "Examples:\n"
        "\n"
        "\b\n"
        "  uhls bind input.sched.uhir\n"
        "\n"
        "\b\n"
        "  uhls bind input.sched.uhir --algo left_edge -o output.bind.uhir\n"
        "\n"
        "\b\n"
        "  uhls bind input.sched.uhir --algo left_edge --flatten -o output.bind.uhir\n"
        "\n"
        "\b\n"
        "  uhls bind input.sched.uhir --algo compat -o output.bind.uhir\n"
        "\n"
        "\b\n"
        "  uhls bind input.bind.uhir --dump conflict\n"
        "\n"
        "\b\n"
        "  uhls bind input.sched.uhir --algo left_edge --dump conflict --dot\n"
        "\n"
        "\b\n"
        "  uhls bind input.bind.uhir --dump trp,trp_unroll --dot --compact\n"
        "\n"
        "\b\n"
        "  uhls bind input.bind.uhir --dump dfgsb,dfgsb_unroll --dot\n"
    ),
)
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--algo",
    help=(
        "Operation binder name. "
        "Use 'left_edge' for concrete interval-based FU+register binding, "
        "or 'compat' for symbolic-schedule FU-only binding."
    ),
)
@click.option(
    "--flatten",
    is_flag=True,
    help="Flatten fully static hierarchical schedules into one global occurrence space before binding. Errors on non-static loop timing.",
)
@click.option(
    "--dump",
    "dump_spec",
    help=(
        "Bind analysis dump(s): "
        + ", ".join(BIND_DUMP_KINDS)
        + ". Pass one kind or a comma-separated list."
    ),
)
@click.option("--dot", is_flag=True, help="Render bind dump(s) as DOT. Without --dump, acts like --dump=conflict.")
@click.option("--compact", is_flag=True, help="Use compact labels for bind dump rendering.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def bind_cmd(
    input_path: Path,
    algo: str | None,
    flatten: bool,
    dump_spec: str | None,
    dot: bool,
    compact: bool,
    output: Path | None,
) -> None:
    """Bind sched-stage µhIR operations to concrete resources."""
    design = parse_uhir_file(input_path)
    if flatten and algo is not None and algo.strip().lower().replace("-", "_") == "compat":
        raise CLIError("'compat' does not support --flatten; use 'left_edge' for fully static flattened binding")

    dump_kinds: tuple[str, ...] = ()
    if dump_spec is not None:
        try:
            dump_kinds = parse_bind_dump_spec(dump_spec)
        except ValueError as exc:
            raise CLIError(str(exc)) from exc
    elif dot:
        dump_kinds = ("conflict",)

    if dump_kinds:
        if design.stage == "bind":
            bound = design
        elif design.stage == "sched":
            if algo is None:
                raise CLIError("'bind' requires --algo when dumping from sched-stage µhIR input")
            bound = lower_sched_to_bind(design, binder=_lookup_operation_binder(algo, binder_kwargs={"flatten": flatten}))
        else:
            raise CLIError(f"'bind --dump' expects sched/bind-stage µhIR input, got stage '{design.stage}'")
        rendered = bind_dump_to_dot(bound, dump_kinds, compact=compact) if dot else format_bind_dump(bound, dump_kinds, compact=compact)
        _write_or_print_text(rendered, output)
        return

    if design.stage != "sched":
        raise CLIError(f"'bind' expects sched-stage µhIR input unless --dump is used, got stage '{design.stage}'")
    bound = lower_sched_to_bind(design, binder=_lookup_operation_binder(algo, binder_kwargs={"flatten": flatten}))
    rendered = format_uhir(bound)
    _write_or_print_text(rendered, output)


@cli.command(
    "fsm",
    help=(
        "Lower bind-stage µhIR to the fsm/FSMD stage.\n"
        "\n"
        "Examples:\n"
        "\n"
        "\b\n"
        "  uhls fsm input.bind.uhir\n"
        "\n"
        "\b\n"
        "  uhls fsm input.bind.uhir --encoding=binary -o output.fsm.uhir\n"
        "\n"
        "\b\n"
        "  uhls fsm input.bind.uhir --encoding=one_hot -o output.fsm.uhir\n"
        "\n"
        "\b\n"
        "  uhls fsm input.bind.uhir --encoding=binary --dot -o output.dot\n"
    ),
)
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--encoding",
    type=click.Choice(FSM_ENCODINGS, case_sensitive=False),
    default="binary",
    show_default=True,
    help="Controller-state encoding.",
)
@click.option("--dot", "emit_dot", is_flag=True, help="Render the synthesized controller as a Graphviz state diagram.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def fsm_cmd(input_path: Path, encoding: str, emit_dot: bool, output: Path | None) -> None:
    """Lower bind-stage µhIR to the fsm/FSMD stage."""
    design = parse_uhir_file(input_path)
    if design.stage != "bind":
        raise CLIError(f"'fsm' expects bind-stage µhIR input, got stage '{design.stage}'")
    lowered = lower_bind_to_fsm(design, encoding=encoding)
    _write_or_print_text(fsm_to_dot(lowered) if emit_dot else format_uhir(lowered), output)


@cli.command(
    "uglir",
    help=(
        "Lower fsm-stage µhIR to the uglir hardware-glue stage.\n"
        "\n"
        "Examples:\n"
        "\n"
        "\b\n"
        "  uhls uglir input.fsm.uhir\n"
        "\n"
        "\b\n"
        "  uhls uglir input.fsm.uhir -o output.uglir\n"
        "\n"
        "\b\n"
        "  uhls uglir input.fsm.uhir --ressources ressources.json -o output.uglir\n"
    ),
)
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--ressources",
    "--resources",
    "-res",
    "resources_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional component-library JSON used to drive uglir instance-port attachment/signature handling.",
)
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
def uglir_cmd(input_path: Path, resources_path: Path | None, output: Path | None) -> None:
    """Lower fsm-stage µhIR to uglir."""
    design = parse_uhir_file(input_path)
    if design.stage != "fsm":
        raise CLIError(f"'uglir' expects fsm-stage µhIR input, got stage '{design.stage}'")
    component_library = _load_component_library(resources_path) if resources_path is not None else None
    lowered = lower_fsm_to_uglir(design, component_library=component_library)
    _write_or_print_text(format_uhir(lowered), output)


@cli.command("hls-emit")
@click.argument("binding")
@click.argument("schedule")
@click.option("-o", "--output")
def hls_emit_cmd(binding: str, schedule: str, output: str | None) -> None:
    """Placeholder for RTL emission flows."""
    del binding, schedule, output
    raise NotImplementedError("'hls-emit' is not implemented yet")


def _load_ir_file(path: Path) -> object:
    return parse_module(path.read_text(encoding="utf-8"))


def _load_executability_graph(path: Path) -> ExecutabilityGraph:
    if path.suffix == ".uhir":
        try:
            return executability_graph_from_uhir(parse_uhir_file(path))
        except (OSError, UHIRParseError, ValueError) as exc:
            raise CLIError(f"failed to load executability graph '{path}': {exc}") from exc

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CLIError(f"failed to load executability graph '{path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise CLIError(f"executability graph '{path}' must be a JSON object")

    if "edges" in payload:
        functional_units = payload.get("functional_units", payload.get("fus"))
        operations = payload.get("operations", payload.get("ops"))
        edges = payload.get("edges")
        if not isinstance(functional_units, list) or not isinstance(operations, list) or not isinstance(edges, list):
            raise CLIError(
                f"executability graph '{path}' must define list-valued 'functional_units'/'fus', 'operations'/'ops', and 'edges'"
            )

        normalized_edges: list[tuple[str, str, int, int]] = []
        for edge in edges:
            if not isinstance(edge, list | tuple) or len(edge) not in {2, 3, 4}:
                raise CLIError(
                    f"executability graph '{path}' has invalid edge {edge!r}; expected [fu, op], [fu, op, weight], or [fu, op, ii, d]"
                )
            if len(edge) == 2:
                ii = 1
                delay = 1
            elif len(edge) == 3:
                weight = edge[2]
                if isinstance(weight, dict):
                    ii = weight.get("ii")
                    delay = weight.get("d")
                    if not isinstance(ii, int) or not isinstance(delay, int):
                        raise CLIError(
                            f"executability graph '{path}' has invalid weight {weight!r}; expected integer ii/d fields"
                        )
                else:
                    ii = 1
                    delay = weight
            else:
                ii = edge[2]
                delay = edge[3]
            if not isinstance(ii, int) or not isinstance(delay, int):
                raise CLIError(f"executability graph '{path}' has non-integer ii/d values {(ii, delay)!r}")
            normalized_edges.append((str(edge[0]), str(edge[1]), ii, delay))

        return ExecutabilityGraph(
            functional_units=tuple(str(vertex) for vertex in functional_units),
            operations=tuple(str(vertex) for vertex in operations),
            edges=tuple(normalized_edges),
        )

    if "functional_units" in payload or "fus" in payload:
        adjacency = payload.get("functional_units", payload.get("fus"))
        if not isinstance(adjacency, dict):
            raise CLIError(f"executability graph '{path}' must use an object for 'functional_units'/'fus' adjacency")
        return ExecutabilityGraph.from_mapping(adjacency)

    if "components" in payload:
        components = _load_component_library(path)

        functional_units: list[str] = []
        operation_order: dict[str, None] = {}
        normalized_edges: list[tuple[str, str, int, int]] = []

        for component_name, component_payload in components.items():
            if not isinstance(component_payload, dict):
                raise CLIError(
                    f"executability graph '{path}' component '{component_name}' must be a JSON object"
                )
            supports = component_payload.get("supports")
            if not isinstance(supports, dict):
                raise CLIError(
                    f"executability graph '{path}' component '{component_name}' must define object-valued 'supports'"
                )

            functional_units.append(str(component_name))
            for operation, weight in supports.items():
                operation_name = str(operation)
                operation_order.setdefault(operation_name, None)
                if not isinstance(weight, dict):
                    raise CLIError(
                        f"executability graph '{path}' component '{component_name}' support '{operation_name}' must be an object"
                    )
                ii = weight.get("ii")
                delay = weight.get("d")
                if not isinstance(ii, int) or not isinstance(delay, int):
                    raise CLIError(
                        f"executability graph '{path}' component '{component_name}' support '{operation_name}' must define integer ii/d"
                    )
                normalized_edges.append((str(component_name), operation_name, ii, delay))

        return ExecutabilityGraph(
            functional_units=tuple(functional_units),
            operations=tuple(operation_order),
            edges=tuple(normalized_edges),
        )

    raise CLIError(
        f"executability graph '{path}' must define either explicit vertices+edges, a FU adjacency object, or a component library"
    )


def _load_component_library(path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CLIError(f"failed to load component library '{path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise CLIError(f"component library '{path}' must be a JSON object")
    components = payload.get("components")
    if not isinstance(components, dict):
        raise CLIError(f"component library '{path}' must define object-valued 'components'")
    normalized: dict[str, dict[str, object]] = {}
    for component_name, component_payload in components.items():
        if not isinstance(component_payload, dict):
            raise CLIError(f"component library '{path}' component '{component_name}' must be a JSON object")
        normalized[str(component_name)] = component_payload
    return normalized


def _format_executability_graph_json(graph: ExecutabilityGraph) -> str:
    payload = {
        "functional_units": list(graph.functional_units),
        "operations": list(graph.operations),
        "edges": [[source, target, {"ii": ii, "d": delay}] for source, target, ii, delay in graph.edges],
    }
    return json.dumps(payload, indent=2)


def _write_or_print_text(text: str, output_path: Path | None) -> None:
    rendered = f"{text}\n" if text and not text.endswith("\n") else text
    if output_path is None:
        if rendered:
            click.echo(rendered, nl=False)
        return
    output_path.write_text(rendered, encoding="utf-8")


def _selected_functions(module: object, requested: str | None) -> list[object]:
    functions = list(getattr(module, "functions", []))
    if requested is None:
        return functions
    selected = getattr(module, "get_function")(requested)
    if selected is None:
        raise CLIError(f"unknown function '{requested}'")
    return [selected]


def _select_function_for_run(module: object, requested: str | None) -> object:
    if requested is not None:
        selected = getattr(module, "get_function")(requested)
        if selected is None:
            raise CLIError(f"unknown function '{requested}'")
        return selected

    functions = list(getattr(module, "functions", []))
    if len(functions) == 1:
        return functions[0]
    main_function = getattr(module, "get_function")("main")
    if main_function is not None:
        return main_function
    raise CLIError("module contains multiple functions; pass --function")


def _format_cfg_summary(function: object) -> str:
    cfg = build_cfg(function)
    flow = control_flow(function)
    lines = [f"func {getattr(function, 'name')}"]
    for label in cfg.order:
        preds = ", ".join(sorted(flow.predecessors[label])) or "-"
        succs = ", ".join(sorted(flow.successors[label])) or "-"
        lines.append(f"  {label}: preds=[{preds}] succs=[{succs}]")
    return "\n".join(lines)


def _render_dfg_dot(function: object, requested_block: str | None, compact: bool = False) -> str:
    info = build_dfg(function)
    if requested_block is None:
        return to_dfg_dot(info, compact=compact)
    graph = info.blocks.get(requested_block)
    if graph is None:
        raise CLIError(f"unknown block '{requested_block}' in function '{getattr(function, 'name')}'")
    return to_basic_block_dfg_dot(graph, compact=compact)


def _format_dfg_summary(function: object, requested_block: str | None) -> str:
    info = build_dfg(function)
    lines = [f"func {getattr(function, 'name')}"]
    block_labels = [requested_block] if requested_block is not None else [block.label for block in function.blocks]
    for label in block_labels:
        graph = info.blocks.get(label)
        if graph is None:
            raise CLIError(f"unknown block '{label}' in function '{getattr(function, 'name')}'")
        lines.append(f"  block {label}: nodes={len(graph.nodes)} edges={len(graph.edges)}")
        for edge in graph.edges:
            lines.append(f"    {edge.source} -> {edge.target} [{edge.kind}:{edge.label}]")
    return "\n".join(lines)


def _format_cdfg_summary(function: object) -> str:
    cfg = build_cfg(function)
    dfg = build_dfg(function)
    control_edges = sum(len(cfg.successors[label]) for label in cfg.order)
    lines = [f"func {getattr(function, 'name')}: blocks={len(cfg.order)} control_edges={control_edges}"]
    for block in function.blocks:
        graph = dfg.blocks[block.label]
        lines.append(f"  block {block.label}: dfg_nodes={len(graph.nodes)} dfg_edges={len(graph.edges)}")
    return "\n".join(lines)


def _parse_run_argument_items(items: list[str]) -> dict[str, str]:
    arguments: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise CLIError(f"invalid --arg value {item!r}; expected name=value")
        name, value_text = item.split("=", 1)
        arguments[name.strip()] = value_text.strip()
    return arguments


def _parse_run_arguments(
    function: object,
    raw_arguments: dict[str, str],
) -> tuple[dict[str, int], dict[str, dict[str, object]]]:
    parameter_types = {
        getattr(parameter, "name"): normalize_type(getattr(parameter, "type", None))
        for parameter in getattr(function, "params", [])
    }
    scalar_arguments: dict[str, int] = {}
    array_arguments: dict[str, dict[str, object]] = {}

    for name, payload in raw_arguments.items():
        parameter_type = parameter_types.get(name)
        if isinstance(parameter_type, ArrayType):
            array_arguments[name] = _parse_array_argument(name, payload)
            continue
        scalar_arguments[name] = _parse_scalar_argument(name, payload)

    return scalar_arguments, array_arguments


def _parse_scalar_argument(name: str, payload: str) -> int:
    try:
        return int(payload)
    except ValueError as exc:
        raise CLIError(
            f"invalid scalar argument for '{name}': {payload!r}; expected integer like --arg {name}=7"
        ) from exc


def _parse_array_argument(name: str, payload: str) -> dict[str, object]:
    element_type = "i32"
    values_text = payload

    if payload.startswith("["):
        close_index = payload.find("]")
        if close_index == -1:
            raise CLIError(
                f"invalid array argument for '{name}': {payload!r}; expected --arg {name}=[v1,v2,...][:type]"
            )
        values_text = payload[1:close_index]
        remainder = payload[close_index + 1 :].strip()
        if remainder:
            if not remainder.startswith(":"):
                raise CLIError(
                    f"invalid array argument for '{name}': {payload!r}; expected --arg {name}=[v1,v2,...][:type]"
                )
            element_type = remainder[1:].strip() or "i32"
    elif ":" in payload:
        values_text, element_type = payload.rsplit(":", 1)

    try:
        data = [] if not values_text.strip() else [int(part.strip()) for part in values_text.split(",")]
    except ValueError as exc:
        raise CLIError(
            f"invalid array argument for '{name}': {payload!r}; expected integers like --arg {name}=[1,2,3]:i32"
        ) from exc
    return {"data": data, "element_type": element_type.strip() or "i32"}


def _parse_scheduler_cli_kwargs(*, sgu_latency_max: str | None) -> dict[str, object]:
    scheduler_kwargs: dict[str, object] = {}
    if sgu_latency_max is not None:
        scheduler_kwargs["sgu_latency_max"] = _parse_sgu_latency_max_spec(sgu_latency_max)
    return scheduler_kwargs


def _parse_sgu_latency_max_spec(spec_text: str) -> dict[str, object]:
    text = spec_text.strip()
    if not text:
        raise CLIError("--sgu_latency_max must not be empty")
    if text.startswith("asap"):
        slack = 0
        remainder = text[4:]
        if remainder:
            if not remainder.startswith("+"):
                raise CLIError("--sgu_latency_max=asap only supports an optional non-negative '+slack' suffix")
            try:
                slack = int(remainder[1:])
            except ValueError as exc:
                raise CLIError("--sgu_latency_max=asap+<slack> requires one integer slack value") from exc
            if slack < 0:
                raise CLIError("--sgu_latency_max=asap+<slack> requires slack >= 0")
        return {"mode": "asap", "slack": slack}

    values: dict[str, int] = {}
    for item in text.split(","):
        entry = item.strip()
        if not entry:
            raise CLIError("--sgu_latency_max region list must not contain empty entries")
        if ":" not in entry:
            raise CLIError("--sgu_latency_max region list must use region:value entries")
        region_name, value_text = entry.split(":", 1)
        region_name = region_name.strip()
        value_text = value_text.strip()
        if not region_name:
            raise CLIError("--sgu_latency_max region list must use non-empty region names")
        try:
            value = int(value_text)
        except ValueError as exc:
            raise CLIError(f"--sgu_latency_max region '{region_name}' must map to an integer latency") from exc
        if value < 0:
            raise CLIError(f"--sgu_latency_max region '{region_name}' must use latency >= 0")
        values[region_name] = value
    return {"mode": "explicit", "values": values}


def _validate_run_invocation(
    function: object,
    scalar_arguments: dict[str, int],
    array_arguments: dict[str, dict[str, object]],
) -> None:
    function_name = getattr(function, "name", "<anon>")
    scalar_parameters: set[str] = set()
    array_parameters: set[str] = set()

    for parameter in getattr(function, "params", []):
        name = getattr(parameter, "name")
        parameter_type = normalize_type(getattr(parameter, "type", None))
        if isinstance(parameter_type, ArrayType):
            array_parameters.add(name)
        else:
            scalar_parameters.add(name)

    unexpected_scalars = sorted(name for name in scalar_arguments if name not in scalar_parameters | array_parameters)
    unexpected_arrays = sorted(name for name in array_arguments if name not in scalar_parameters | array_parameters)
    unexpected_arguments = sorted(set(unexpected_scalars) | set(unexpected_arrays))
    if unexpected_arguments:
        raise CLIError(f"unknown arguments for function '{function_name}': {', '.join(unexpected_arguments)}")

    missing_scalars = sorted(name for name in scalar_parameters if name not in scalar_arguments)
    if missing_scalars:
        raise CLIError(
            f"missing scalar arguments for function '{function_name}': "
            f"{', '.join(missing_scalars)}; pass --arg name=value"
        )

    missing_arrays = sorted(name for name in array_parameters if name not in array_arguments)
    if missing_arrays:
        raise CLIError(
            f"missing array arguments for function '{function_name}': "
            f"{', '.join(missing_arrays)}; pass --arg name=[v1,v2,...][:type]"
        )


def _lookup_flat_sgu_scheduler(name: str | None, scheduler_kwargs: dict[str, object] | None = None) -> object:
    spec_text = "asap" if name is None else name.strip()
    effective_scheduler_kwargs = {} if scheduler_kwargs is None else dict(scheduler_kwargs)
    if _looks_like_external_python_symbol_spec(spec_text):
        return _load_external_flat_sgu_scheduler(spec_text, effective_scheduler_kwargs)
    try:
        return create_builtin_scheduler(spec_text, **effective_scheduler_kwargs)
    except ValueError as exc:
        supported = ", ".join(builtin_scheduler_names())
        raise CLIError(f"unknown flat SGU scheduler '{spec_text}'; expected one of: {supported} or /path/to/scheduler.py:Symbol") from exc


def _lookup_operation_binder(name: str | None, binder_kwargs: dict[str, object] | None = None) -> object:
    spec_text = "left_edge" if name is None else name.strip()
    effective_binder_kwargs = {} if binder_kwargs is None else dict(binder_kwargs)
    try:
        return create_builtin_binder(spec_text, **effective_binder_kwargs)
    except ValueError as exc:
        supported = ", ".join(builtin_binder_names())
        raise CLIError(f"unknown operation binder '{spec_text}'; expected one of: {supported}") from exc


def _lookup_gopt_pass(name: str, pass_args: tuple[str, ...] = ()) -> object:
    if _looks_like_external_python_symbol_spec(name):
        return _load_external_gopt_pass(name, pass_args)
    normalized = name.strip().lower().replace("-", "_")
    try:
        return _gopt_specs_by_cli_name()[normalized].factory()
    except KeyError as exc:
        supported = ", ".join(builtin_gopt_pass_names())
        raise CLIError(f"unknown graph optimization pass '{name}'; expected one of: {supported}") from exc


def _lookup_opt_pass(name: str, pass_args: tuple[str, ...] = ()) -> object:
    if _looks_like_external_python_symbol_spec(name):
        return _load_external_opt_pass(name, pass_args)
    normalized = name.strip().lower().replace("-", "_")
    try:
        return _opt_specs_by_cli_name()[normalized].factory()
    except KeyError as exc:
        raise CLIError(f"unknown optimization pass '{name}'") from exc


def _looks_like_external_python_symbol_spec(name: str) -> bool:
    head = name.strip().split(":", 1)[0].strip()
    return head.endswith(".py")


def _load_external_flat_sgu_scheduler(spec_text: str, scheduler_kwargs: dict[str, object] | None = None) -> object:
    path_text, symbol_name = _split_external_symbol_spec(
        spec_text,
        kind="flat SGU scheduler",
        syntax="/path/to/scheduler.py:Symbol",
    )
    module_path = Path(path_text).expanduser()
    if not module_path.is_absolute():
        module_path = Path.cwd() / module_path
    module_path = module_path.resolve()
    if not module_path.is_file():
        raise CLIError(f"external flat SGU scheduler file does not exist: '{module_path}'")

    module = _load_external_python_module(module_path, _external_python_module_name("sched", module_path), "flat SGU scheduler")
    try:
        target = getattr(module, symbol_name)
    except AttributeError as exc:
        raise CLIError(f"external flat SGU scheduler '{spec_text}' is missing symbol '{symbol_name}'") from exc
    return _materialize_external_flat_sgu_scheduler(target, spec_text, {} if scheduler_kwargs is None else scheduler_kwargs)


def _load_external_opt_pass(spec_text: str, pass_args: tuple[str, ...] = ()) -> object:
    path_text, symbol_name = _split_external_symbol_spec(
        spec_text,
        kind="external optimization pass",
        syntax="/path/to/pass.py:Symbol",
    )
    module_path = Path(path_text).expanduser()
    if not module_path.is_absolute():
        module_path = Path.cwd() / module_path
    module_path = module_path.resolve()
    if not module_path.is_file():
        module_path = _remap_legacy_external_opt_path(module_path)
    if not module_path.is_file():
        raise CLIError(f"external optimization pass file does not exist: '{module_path}'")

    module = _load_external_python_module(
        module_path,
        _external_python_module_name("opt", module_path),
        "optimization pass",
    )
    try:
        target = getattr(module, symbol_name)
    except AttributeError as exc:
        raise CLIError(
            f"external optimization pass '{spec_text}' is missing symbol '{symbol_name}'"
        ) from exc
    return _materialize_external_opt_pass(target, spec_text, pass_args)


def _load_external_gopt_pass(spec_text: str, pass_args: tuple[str, ...] = ()) -> object:
    path_text, symbol_name = _split_external_symbol_spec(
        spec_text,
        kind="external graph optimization pass",
        syntax="/path/to/pass.py:Symbol",
    )
    module_path = Path(path_text).expanduser()
    if not module_path.is_absolute():
        module_path = Path.cwd() / module_path
    module_path = module_path.resolve()
    if not module_path.is_file():
        raise CLIError(f"external graph optimization pass file does not exist: '{module_path}'")

    module = _load_external_python_module(
        module_path,
        _external_python_module_name("gopt", module_path),
        "graph optimization pass",
    )
    try:
        target = getattr(module, symbol_name)
    except AttributeError as exc:
        raise CLIError(
            f"external graph optimization pass '{spec_text}' is missing symbol '{symbol_name}'"
        ) from exc
    return _materialize_external_gopt_pass(target, spec_text, pass_args)


def _split_external_symbol_spec(spec_text: str, *, kind: str, syntax: str) -> tuple[str, str]:
    raw = spec_text.strip()
    if ":" not in raw:
        raise CLIError(f"{kind} '{spec_text}' must use '{syntax}'")
    path_text, symbol_name = raw.rsplit(":", 1)
    if not path_text.strip() or not symbol_name.strip():
        raise CLIError(f"{kind} '{spec_text}' must use '{syntax}'")
    return path_text.strip(), symbol_name.strip()


def _remap_legacy_external_opt_path(module_path: Path) -> Path:
    path_text = str(module_path)
    legacy_rewrites = (
        ("/src/uhls/passes/", "/src/uhls/middleend/passes/"),
        ("/src/uhls/ir/", "/src/uhls/middleend/uir/"),
    )
    for old, new in legacy_rewrites:
        if old in path_text:
            return Path(path_text.replace(old, new))
    return module_path


def _external_python_module_name(kind: str, path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    stem = "".join(char if char.isalnum() or char == "_" else "_" for char in path.stem)
    return f"uhls_external_{kind}_{stem}_{digest}"


def _load_external_python_module(module_path: Path, module_name: str, kind: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise CLIError(f"could not load external {kind} module from '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise CLIError(f"failed to import external {kind} module '{module_path}': {exc}") from exc
    return module


def _materialize_external_flat_sgu_scheduler(target: object, spec_text: str, scheduler_kwargs: dict[str, object]) -> object:
    if isinstance(target, type):
        return _instantiate_external_flat_sgu_scheduler_class(target, spec_text, scheduler_kwargs)
    if hasattr(target, "schedule_sgu"):
        if scheduler_kwargs:
            raise CLIError(
                f"external flat SGU scheduler '{spec_text}' is already instantiated and cannot consume scheduler arguments"
            )
        return target
    if callable(target):
        return _materialize_external_flat_sgu_scheduler_callable(target, spec_text, scheduler_kwargs)
    raise CLIError(
        f"external flat SGU scheduler '{spec_text}' must resolve to a scheduler object, callable, or no-arg scheduler class"
    )


def _instantiate_external_flat_sgu_scheduler_class(target: type, spec_text: str, scheduler_kwargs: dict[str, object]) -> object:
    try:
        params = list(signature(target).parameters.values())
    except (TypeError, ValueError):
        params = []

    positional = [
        param
        for param in params
        if param.kind in {InspectParameter.POSITIONAL_ONLY, InspectParameter.POSITIONAL_OR_KEYWORD}
    ]
    required = [param for param in positional if param.default is InspectParameter.empty]
    try:
        if scheduler_kwargs:
            try:
                instance = target(**scheduler_kwargs)
            except TypeError:
                if len(required) <= 1 and len(positional) <= 1:
                    instance = target(scheduler_kwargs)
                else:
                    raise
        elif not required:
            instance = target()
        elif len(required) <= 1 and len(positional) <= 1:
            instance = target(())
        else:
            raise CLIError(
                f"external flat SGU scheduler '{spec_text}' must be instantiable with no arguments, one scheduler_args tuple/dict, or keyword scheduler arguments"
            )
    except TypeError as exc:
        raise CLIError(
            f"external flat SGU scheduler '{spec_text}' must be instantiable with no arguments, one scheduler_args tuple/dict, or keyword scheduler arguments"
        ) from exc
    if hasattr(instance, "schedule_sgu"):
        return instance
    raise CLIError(f"external flat SGU scheduler '{spec_text}' class must provide schedule_sgu(region)")


def _materialize_external_flat_sgu_scheduler_callable(
    target: Callable[..., object],
    spec_text: str,
    scheduler_kwargs: dict[str, object],
) -> object:
    try:
        params = list(signature(target).parameters.values())
    except (TypeError, ValueError):
        if scheduler_kwargs:
            raise CLIError(
                f"external flat SGU scheduler '{spec_text}' with scheduler arguments must be a class or keyword-accepting factory"
            )
        return CallableSGUScheduler(target)  # type: ignore[arg-type]

    positional = [
        param
        for param in params
        if param.kind in {InspectParameter.POSITIONAL_ONLY, InspectParameter.POSITIONAL_OR_KEYWORD}
    ]
    required = [param for param in positional if param.default is InspectParameter.empty]
    if scheduler_kwargs:
        keyword_capable = any(param.kind == InspectParameter.VAR_KEYWORD for param in params) or all(
            key in {param.name for param in params if param.kind in {InspectParameter.KEYWORD_ONLY, InspectParameter.POSITIONAL_OR_KEYWORD}}
            for key in scheduler_kwargs
        )
        if keyword_capable:
            try:
                produced = target(**scheduler_kwargs)
            except TypeError as exc:
                raise CLIError(
                    f"external flat SGU scheduler factory '{spec_text}' could not be called with scheduler arguments"
                ) from exc
            return _materialize_external_flat_sgu_scheduler(produced, spec_text, {})
        raise CLIError(
            f"external flat SGU scheduler '{spec_text}' with scheduler arguments must be a class or keyword-accepting factory"
        )
    if len(required) <= 1 and len(positional) <= 1:
        if len(positional) == 1:
            return CallableSGUScheduler(target)  # type: ignore[arg-type]
        try:
            produced = target()
        except TypeError as exc:
            raise CLIError(
                f"external flat SGU scheduler factory '{spec_text}' could not be called with no arguments"
            ) from exc
        return _materialize_external_flat_sgu_scheduler(produced, spec_text)
    raise CLIError(
        f"external flat SGU scheduler '{spec_text}' must be a schedule_sgu(region) callable or no-arg factory"
    )


def _materialize_external_opt_pass(target: object, spec_text: str, pass_args: tuple[str, ...]) -> object:
    if isinstance(target, type):
        return _instantiate_external_pass_class(target, spec_text, pass_args)
    if hasattr(target, "run"):
        return target if hasattr(target, "name") else _NamedPassAdapter(spec_text, target)
    if callable(target):
        materialized = _maybe_call_external_pass_factory(target, spec_text, pass_args)
        return materialized
    raise CLIError(
        f"external optimization pass '{spec_text}' must resolve to a pass object, callable, or no-arg pass class"
    )


def _materialize_external_gopt_pass(target: object, spec_text: str, pass_args: tuple[str, ...]) -> object:
    if isinstance(target, type):
        return _instantiate_external_gopt_pass_class(target, spec_text, pass_args)
    if hasattr(target, "run"):
        return target if hasattr(target, "name") else _NamedPassAdapter(spec_text, target)
    if callable(target):
        return _maybe_call_external_gopt_pass_factory(target, spec_text, pass_args)
    raise CLIError(
        f"external graph optimization pass '{spec_text}' must resolve to a pass object, callable, or no-arg pass class"
    )


def _instantiate_external_pass_class(target: type, spec_text: str, pass_args: tuple[str, ...]) -> object:
    try:
        params = list(signature(target).parameters.values())
    except (TypeError, ValueError):
        params = []

    positional = [
        param
        for param in params
        if param.kind in {InspectParameter.POSITIONAL_ONLY, InspectParameter.POSITIONAL_OR_KEYWORD}
    ]
    required = [param for param in positional if param.default is InspectParameter.empty]
    try:
        if not required and not positional:
            instance = target()
            return instance if hasattr(instance, "name") else _NamedPassAdapter(spec_text, instance)
        if len(required) <= 1 and len(positional) <= 1:
            instance = target(tuple(pass_args))
            return instance if hasattr(instance, "name") else _NamedPassAdapter(spec_text, instance)
    except TypeError as exc:
        raise CLIError(
            f"external optimization pass '{spec_text}' must be instantiable with no arguments or one pass_args tuple"
        ) from exc
    raise CLIError(
        f"external optimization pass '{spec_text}' must be instantiable with no arguments or one pass_args tuple"
    )


def _instantiate_external_gopt_pass_class(target: type, spec_text: str, pass_args: tuple[str, ...]) -> object:
    try:
        params = list(signature(target).parameters.values())
    except (TypeError, ValueError):
        params = []

    positional = [
        param
        for param in params
        if param.kind in {InspectParameter.POSITIONAL_ONLY, InspectParameter.POSITIONAL_OR_KEYWORD}
    ]
    required = [param for param in positional if param.default is InspectParameter.empty]
    try:
        if not required and not positional:
            instance = target()
            return instance if hasattr(instance, "name") else _NamedPassAdapter(spec_text, instance)
        if len(required) <= 1 and len(positional) <= 1:
            instance = target(tuple(pass_args))
            return instance if hasattr(instance, "name") else _NamedPassAdapter(spec_text, instance)
    except TypeError as exc:
        raise CLIError(
            f"external graph optimization pass '{spec_text}' must be instantiable with no arguments or one pass_args tuple"
        ) from exc
    raise CLIError(
        f"external graph optimization pass '{spec_text}' must be instantiable with no arguments or one pass_args tuple"
    )


def _maybe_call_external_pass_factory(
    target: Callable[..., object], spec_text: str, pass_args: tuple[str, ...]
) -> object:
    try:
        params = list(signature(target).parameters.values())
    except (TypeError, ValueError):
        return target

    positional = [
        param
        for param in params
        if param.kind in {InspectParameter.POSITIONAL_ONLY, InspectParameter.POSITIONAL_OR_KEYWORD}
    ]
    required = [param for param in positional if param.default is InspectParameter.empty]
    if required or positional:
        return target

    try:
        produced = target()
    except TypeError as exc:
        raise CLIError(
            f"external optimization pass factory '{spec_text}' could not be called with no arguments"
        ) from exc
    if callable(produced) or hasattr(produced, "run"):
        return produced
    raise CLIError(
        f"external optimization pass factory '{spec_text}' must return a pass object or callable"
    )


def _maybe_call_external_gopt_pass_factory(
    target: Callable[..., object], spec_text: str, pass_args: tuple[str, ...]
) -> object:
    try:
        params = list(signature(target).parameters.values())
    except (TypeError, ValueError):
        return target

    positional = [
        param
        for param in params
        if param.kind in {InspectParameter.POSITIONAL_ONLY, InspectParameter.POSITIONAL_OR_KEYWORD}
    ]
    required = [param for param in positional if param.default is InspectParameter.empty]
    if required or positional:
        return target

    try:
        produced = target()
    except TypeError as exc:
        raise CLIError(
            f"external graph optimization pass factory '{spec_text}' could not be called with no arguments"
        ) from exc
    if callable(produced) or hasattr(produced, "run"):
        return produced
    raise CLIError(
        f"external graph optimization pass factory '{spec_text}' must return a pass object or callable"
    )


def _invoke_external_pass_run(delegate: object, ir: object, context: PassContext) -> object:
    run = getattr(delegate, "run", None)
    if not callable(run):
        raise TypeError(f"unsupported external pass object {delegate!r}")
    params = list(signature(run).parameters.values())
    positional = [
        param
        for param in params
        if param.kind in {InspectParameter.POSITIONAL_ONLY, InspectParameter.POSITIONAL_OR_KEYWORD}
    ]
    required = [param for param in positional if param.default is InspectParameter.empty]
    if len(required) <= 1 and len(positional) <= 1:
        return run(ir)
    if len(required) <= 2 and len(positional) <= 2:
        return run(ir, context)
    if len(required) <= 3 and len(positional) <= 3:
        return run(ir, context, context.pass_args)
    raise TypeError(
        f"external pass object '{type(delegate).__name__}' must expose run(ir), run(ir, context), or run(ir, context, pass_args)"
    )
