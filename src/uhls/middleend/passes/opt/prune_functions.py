"""Unreachable-function pruning for canonical µhLS IR."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from uhls.middleend.uir import CallOp, Function, Module
from uhls.middleend.passes.util.pass_manager import PassContext


def prune_functions_function(function: Function) -> Function:
    """Leave one standalone function unchanged."""
    return deepcopy(function)


def prune_functions_module(
    module: Module,
    context: PassContext | None = None,
    pass_args: tuple[str, ...] = (),
) -> Module:
    """Remove unreachable functions from one module."""
    del context
    result = deepcopy(module)
    function_map = result.function_map()
    if not function_map:
        return result

    callees_by_function = {
        function.name: _called_internal_functions(function, function_map)
        for function in result.functions
    }
    roots = _root_functions(result, pass_args)
    reachable = _reachable_functions(roots, callees_by_function)
    result.functions = [function for function in result.functions if function.name in reachable]
    return result


@dataclass(frozen=True)
class _PruneFunctionsPass:
    name: str = "prune_functions"

    def run(self, ir: Module, context: PassContext, pass_args: tuple[str, ...]) -> Module:
        return prune_functions_module(ir, context, pass_args)


def PruneFunctionsPass():
    """Return a pass-manager compatible unreachable-function pruning pass wrapper."""
    return _PruneFunctionsPass()


def _called_internal_functions(function: Function, function_map: dict[str, Function]) -> set[str]:
    called: set[str] = set()
    for block in function.blocks:
        for instruction in block.instructions:
            if isinstance(instruction, CallOp) and instruction.callee in function_map:
                called.add(instruction.callee)
    return called


def _root_functions(module: Module, pass_args: tuple[str, ...]) -> set[str]:
    explicit_roots = {name for name in pass_args if name in module.function_map()}
    if explicit_roots:
        return explicit_roots

    main_function = module.get_function("main")
    if main_function is not None:
        return {"main"}

    if module.functions:
        return {module.functions[-1].name}
    return set()


def _reachable_functions(roots: set[str], callees_by_function: dict[str, set[str]]) -> set[str]:
    reachable = set(roots)
    worklist = list(sorted(roots))
    while worklist:
        function_name = worklist.pop()
        for callee in callees_by_function.get(function_name, set()):
            if callee not in reachable:
                reachable.add(callee)
                worklist.append(callee)
    return reachable
