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

from uhls.backend.uhir import (
    ExecutabilityGraph,
    UHIRParseError,
    dummy_executability_graph,
    executability_graph_from_uhir,
    format_uhir,
    lower_seq_to_alloc,
    lower_module_to_seq,
    parse_uhir_file,
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


def main(argv: list[str] | None = None) -> int:
    """Run the µhLS CLI and return a process exit code."""
    active_argv = list(sys.argv[1:] if argv is None else argv)
    opt_help_exit = _maybe_handle_opt_pass_help(active_argv)
    if opt_help_exit is not None:
        return opt_help_exit

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
        "Run optimization pass pipelines.\n\n"
        f"Implemented passes: {_implemented_opt_help()}.\n"
        f"Registered pass names: {', '.join(spec.name for spec in _OPT_PASS_SPECS)}.\n"
        "External pass syntax: /path/to/pass.py:Symbol\n\n"
        "Example: uhls opt input.uir -p simplify_cfg,inline_calls --pass-arg dot4 -o output.uir"
    ),
)
@click.argument("input_path", metavar="input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-p",
    "--passes",
    required=True,
    help=(
        "Comma-separated optimization passes. "
        f"Implemented: {_implemented_opt_help()}. "
        f"Registered: {', '.join(spec.name for spec in _OPT_PASS_SPECS)}. "
        "External: /path/to/pass.py:Symbol"
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
    "run",
    help=(
        "Execute IR with the interpreter.\n\n"
        "Examples: uhls run input.uir --function add1 --arg x=7; "
        "uhls run input.uir --function dot4 --arg A=[1,2,3,4] --arg B=[4,3,2,1]"
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


@cli.command("hls-sched")
@click.argument("input_path", metavar="input")
@click.option("--algo")
@click.option("--resources")
@click.option("-o", "--output")
def hls_sched_cmd(input_path: str, algo: str | None, resources: str | None, output: str | None) -> None:
    """Placeholder for scheduling flows."""
    del input_path, algo, resources, output
    raise NotImplementedError("'hls-sched' is not implemented yet")


@cli.command("hls-bind")
@click.argument("input_path", metavar="input")
@click.option("-o", "--output")
def hls_bind_cmd(input_path: str, output: str | None) -> None:
    """Placeholder for binding flows."""
    del input_path, output
    raise NotImplementedError("'hls-bind' is not implemented yet")


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

    raise CLIError(
        f"executability graph '{path}' must define either explicit vertices+edges or a FU adjacency object"
    )


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

def _lookup_opt_pass(name: str, pass_args: tuple[str, ...] = ()) -> object:
    if _looks_like_external_opt_pass(name):
        return _load_external_opt_pass(name, pass_args)
    normalized = name.strip().lower().replace("-", "_")
    try:
        return _opt_specs_by_cli_name()[normalized].factory()
    except KeyError as exc:
        raise CLIError(f"unknown optimization pass '{name}'") from exc


def _looks_like_external_opt_pass(name: str) -> bool:
    head = name.strip().split(":", 1)[0].strip()
    return head.endswith(".py")


def _load_external_opt_pass(spec_text: str, pass_args: tuple[str, ...] = ()) -> object:
    path_text, symbol_name = _split_external_opt_pass_spec(spec_text)
    module_path = Path(path_text).expanduser()
    if not module_path.is_absolute():
        module_path = Path.cwd() / module_path
    module_path = module_path.resolve()
    if not module_path.is_file():
        module_path = _remap_legacy_external_opt_path(module_path)
    if not module_path.is_file():
        raise CLIError(f"external optimization pass file does not exist: '{module_path}'")

    module_name = _external_opt_module_name(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise CLIError(f"could not load external optimization pass module from '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise CLIError(
            f"failed to import external optimization pass module '{module_path}': {exc}"
        ) from exc
    try:
        target = getattr(module, symbol_name)
    except AttributeError as exc:
        raise CLIError(
            f"external optimization pass '{spec_text}' is missing symbol '{symbol_name}'"
        ) from exc
    return _materialize_external_opt_pass(target, spec_text, pass_args)


def _split_external_opt_pass_spec(spec_text: str) -> tuple[str, str]:
    raw = spec_text.strip()
    if ":" not in raw:
        raise CLIError(
            f"external optimization pass '{spec_text}' must use '/path/to/pass.py:Symbol'"
        )
    path_text, symbol_name = raw.rsplit(":", 1)
    if not path_text.strip() or not symbol_name.strip():
        raise CLIError(
            f"external optimization pass '{spec_text}' must use '/path/to/pass.py:Symbol'"
        )
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


def _external_opt_module_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    stem = "".join(char if char.isalnum() or char == "_" else "_" for char in path.stem)
    return f"uhls_external_opt_{stem}_{digest}"


def _materialize_external_opt_pass(target: object, spec_text: str, pass_args: tuple[str, ...]) -> object:
    if isinstance(target, type):
        return _instantiate_external_pass_class(target, spec_text, pass_args)
    if hasattr(target, "run"):
        return target
    if callable(target):
        materialized = _maybe_call_external_pass_factory(target, spec_text, pass_args)
        return materialized
    raise CLIError(
        f"external optimization pass '{spec_text}' must resolve to a pass object, callable, or no-arg pass class"
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
            return target()
        if len(required) <= 1 and len(positional) <= 1:
            return target(tuple(pass_args))
    except TypeError as exc:
        raise CLIError(
            f"external optimization pass '{spec_text}' must be instantiable with no arguments or one pass_args tuple"
        ) from exc
    raise CLIError(
        f"external optimization pass '{spec_text}' must be instantiable with no arguments or one pass_args tuple"
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
