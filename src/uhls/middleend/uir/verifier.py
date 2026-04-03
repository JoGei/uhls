"""Well-formedness checks for canonical µhLS IR."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .function import Function
from .module import Module
from .types import ArrayType, normalize_type
from .types import type_name
from .values import IncomingValue, Literal, Parameter, Variable

_TERMINATORS = frozenset({"br", "cbr", "ret"})
_VALUE_OPS = frozenset(
    {
        "const",
        "mov",
        "neg",
        "not",
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
        "load",
        "phi",
        "call",
    }
)
_COMPARE_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


class IRVerificationError(ValueError):
    """Raised when a canonical IR object is malformed."""


def verify_module(
    module: Module,
    *,
    require_ssa: bool | None = None,
    allow_calls: bool = True,
    allow_param: bool = False,
) -> Module:
    """Verify all functions in one module and return the module."""
    seen: set[str] = set()
    for function in module.functions:
        if function.name in seen:
            raise IRVerificationError(f"duplicate function '{function.name}'")
        seen.add(function.name)
        verify_function(
            function,
            require_ssa=require_ssa,
            allow_calls=allow_calls,
            allow_param=allow_param,
        )
    return module


def verify_function(
    function: Function,
    *,
    require_ssa: bool | None = None,
    allow_calls: bool = True,
    allow_param: bool = False,
) -> Function:
    """Verify one function against the canonical µhLS IR rules."""
    if not function.blocks:
        raise IRVerificationError(f"function '{function.name}' has no blocks")

    block_map: dict[str, Any] = {}
    for block in function.blocks:
        if not block.label:
            raise IRVerificationError(f"function '{function.name}' has a block with no label")
        if block.label in block_map:
            raise IRVerificationError(f"function '{function.name}' has duplicate block '{block.label}'")
        block_map[block.label] = block

    if function.entry not in block_map:
        raise IRVerificationError(
            f"function '{function.name}' entry block '{function.entry}' does not exist"
        )

    symbol_types: dict[str, Any] = {}
    param_names: set[str] = set()
    for parameter in function.params:
        if parameter.name in param_names:
            raise IRVerificationError(f"function '{function.name}' has duplicate parameter '{parameter.name}'")
        param_names.add(parameter.name)
        symbol_types[parameter.name] = parameter.type
    for local_name, spec in function.local_arrays.items():
        if local_name in symbol_types:
            raise IRVerificationError(
                f"function '{function.name}' has local array '{local_name}' that conflicts with another symbol"
            )
        if not isinstance(spec, dict):
            raise IRVerificationError(
                f"function '{function.name}' has malformed local array declaration for '{local_name}'"
            )
        size = spec.get("size")
        if not isinstance(size, int) or size < 0:
            raise IRVerificationError(
                f"function '{function.name}' has invalid local array size for '{local_name}'"
            )
        element_type = normalize_type(spec.get("element_type"))
        if element_type is None:
            raise IRVerificationError(
                f"function '{function.name}' local array '{local_name}' is missing an element type"
            )
        if isinstance(element_type, ArrayType):
            raise IRVerificationError(
                f"function '{function.name}' local array '{local_name}' must use a scalar element type"
            )
        symbol_types[local_name] = ArrayType(element_type)

    has_phi = any(
        getattr(instruction, "opcode", None) == "phi"
        for block in function.blocks
        for instruction in block.instructions
    )
    ssa_mode = has_phi if require_ssa is None else require_ssa
    defined_names = set(param_names)
    successors: dict[str, set[str]] = {label: set() for label in block_map}

    for block in function.blocks:
        if block.terminator is None:
            raise IRVerificationError(f"block '{block.label}' has no terminator")

        saw_non_phi = False
        for instruction in block.instructions:
            opcode = _opcode(instruction)
            if opcode in _TERMINATORS:
                raise IRVerificationError(
                    f"block '{block.label}' contains terminator '{opcode}' before the end of the block"
                )
            if opcode == "phi":
                if saw_non_phi:
                    raise IRVerificationError(
                        f"block '{block.label}' has phi instructions after non-phi instructions"
                    )
            else:
                saw_non_phi = True
            _verify_instruction(
                instruction,
                function_name=function.name,
                block_label=block.label,
                symbol_types=symbol_types,
                defined_names=defined_names,
                ssa_mode=ssa_mode,
                allow_calls=allow_calls,
                allow_param=allow_param,
            )

        terminator = block.terminator
        opcode = _opcode(terminator)
        if opcode == "br":
            target = str(getattr(terminator, "target"))
            successors[block.label].add(target)
        elif opcode == "cbr":
            cond_type = _operand_type(getattr(terminator, "cond"), symbol_types)
            if cond_type is not None and type_name(cond_type) != "i1":
                raise IRVerificationError(
                    f"cbr in block '{block.label}' must use an i1 condition, got '{type_name(cond_type)}'"
                )
            successors[block.label].add(str(getattr(terminator, "true_target")))
            successors[block.label].add(str(getattr(terminator, "false_target")))
        elif opcode == "ret":
            _verify_return(terminator, function.return_type, symbol_types, block.label)
        else:
            raise IRVerificationError(f"block '{block.label}' has unsupported terminator '{opcode}'")

    for label, targets in successors.items():
        for target in targets:
            if target not in block_map:
                raise IRVerificationError(f"block '{label}' branches to unknown block '{target}'")

    predecessors: dict[str, set[str]] = defaultdict(set)
    for source, targets in successors.items():
        for target in targets:
            predecessors[target].add(source)

    for block in function.blocks:
        expected_preds = predecessors.get(block.label, set())
        for instruction in block.instructions:
            if _opcode(instruction) != "phi":
                break
            incoming = list(getattr(instruction, "incoming"))
            actual_preds = [item.pred if isinstance(item, IncomingValue) else str(item[0]) for item in incoming]
            if len(actual_preds) != len(set(actual_preds)):
                raise IRVerificationError(
                    f"phi '{getattr(instruction, 'dest')}' in block '{block.label}' has duplicate predecessors"
                )
            if set(actual_preds) != expected_preds:
                raise IRVerificationError(
                    f"phi '{getattr(instruction, 'dest')}' in block '{block.label}' does not match CFG predecessors"
                )
            dest_type = getattr(instruction, "type", None)
            for item in incoming:
                value = item.value if isinstance(item, IncomingValue) else item[1]
                incoming_type = _operand_type(value, symbol_types)
                if (
                    incoming_type is not None
                    and dest_type is not None
                    and type_name(incoming_type) != type_name(dest_type)
                ):
                    raise IRVerificationError(
                        f"phi '{getattr(instruction, 'dest')}' in block '{block.label}' has mismatched incoming type"
                    )

    return function


def _verify_instruction(
    instruction: Any,
    *,
    function_name: str,
    block_label: str,
    symbol_types: dict[str, Any],
    defined_names: set[str],
    ssa_mode: bool,
    allow_calls: bool,
    allow_param: bool,
) -> None:
    opcode = _opcode(instruction)

    if opcode == "param" and not allow_param:
        raise IRVerificationError(f"function '{function_name}' uses non-canonical 'param' in block '{block_label}'")

    if opcode == "call" and not allow_calls:
        raise IRVerificationError(f"function '{function_name}' uses disallowed 'call' in block '{block_label}'")

    if opcode not in _VALUE_OPS and opcode not in {"store", "param", "print"}:
        raise IRVerificationError(f"block '{block_label}' uses unsupported opcode '{opcode}'")

    if opcode in _VALUE_OPS:
        dest = getattr(instruction, "dest", None)
        result_type = getattr(instruction, "type", None)
        if not dest:
            raise IRVerificationError(f"opcode '{opcode}' in block '{block_label}' is missing a result name")
        if opcode != "call" or dest is not None:
            if result_type is None:
                raise IRVerificationError(f"opcode '{opcode}' in block '{block_label}' is missing a result type")
        if ssa_mode and dest in defined_names:
            raise IRVerificationError(f"SSA value '{dest}' is defined more than once")
        defined_names.add(dest)
        if result_type is not None:
            symbol_types[dest] = result_type

    if opcode == "const":
        return
    if opcode in {"mov", "neg", "not"}:
        _require_operand(getattr(instruction, "value", None), symbol_types, block_label, opcode)
        return
    if opcode in {
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
    }:
        lhs_type = _require_operand(getattr(instruction, "lhs", None), symbol_types, block_label, opcode)
        rhs_type = _require_operand(getattr(instruction, "rhs", None), symbol_types, block_label, opcode)
        if (
            opcode in _COMPARE_OPS
            and getattr(instruction, "type", None) is not None
            and type_name(getattr(instruction, "type")) != "i1"
        ):
            raise IRVerificationError(f"compare '{opcode}' in block '{block_label}' must produce i1")
        if (
            lhs_type is not None
            and rhs_type is not None
            and type_name(lhs_type) != type_name(rhs_type)
        ):
            raise IRVerificationError(
                f"opcode '{opcode}' in block '{block_label}' uses mismatched operand types"
            )
        return
    if opcode == "load":
        _require_operand(getattr(instruction, "index", None), symbol_types, block_label, opcode)
        return
    if opcode == "store":
        _require_operand(getattr(instruction, "index", None), symbol_types, block_label, opcode)
        _require_operand(getattr(instruction, "value", None), symbol_types, block_label, opcode)
        return
    if opcode == "phi":
        if not getattr(instruction, "incoming", None):
            raise IRVerificationError(f"phi '{getattr(instruction, 'dest')}' in block '{block_label}' has no inputs")
        return
    if opcode == "call":
        for operand in getattr(instruction, "operands", []):
            _require_operand(operand, symbol_types, block_label, opcode)
        return
    if opcode == "print":
        for operand in getattr(instruction, "operands", []):
            _require_operand(operand, symbol_types, block_label, opcode)
        return
    if opcode == "param":
        _require_operand(getattr(instruction, "value", None), symbol_types, block_label, opcode)


def _verify_return(terminator: Any, return_type: Any, symbol_types: dict[str, Any], block_label: str) -> None:
    value = getattr(terminator, "value", None)
    if value is None:
        raise IRVerificationError(f"ret in block '{block_label}' must return a value")
    operand_type = _operand_type(value, symbol_types)
    if operand_type is not None and type_name(operand_type) != type_name(return_type):
        raise IRVerificationError(
            f"ret in block '{block_label}' returns '{type_name(operand_type)}' but function expects '{type_name(return_type)}'"
        )


def _require_operand(operand: Any, symbol_types: dict[str, Any], block_label: str, opcode: str) -> Any:
    if operand is None:
        raise IRVerificationError(f"opcode '{opcode}' in block '{block_label}' is missing an operand")
    return _operand_type(operand, symbol_types)


def _operand_type(operand: Any, symbol_types: dict[str, Any]) -> Any | None:
    if isinstance(operand, Literal):
        return operand.type
    if isinstance(operand, (Parameter, Variable)):
        if operand.type is not None:
            return operand.type
        return symbol_types.get(operand.name)
    if isinstance(operand, bool):
        return "i1"
    if isinstance(operand, int):
        return None
    if isinstance(operand, str):
        if operand not in symbol_types:
            raise IRVerificationError(f"use of undefined value '{operand}'")
        return symbol_types[operand]
    name = getattr(operand, "name", None)
    if name is not None:
        if name not in symbol_types:
            raise IRVerificationError(f"use of undefined value '{name}'")
        return symbol_types[name]
    return getattr(operand, "type", None)


def _opcode(instruction: Any) -> str:
    opcode = getattr(instruction, "opcode", None)
    if not isinstance(opcode, str):
        raise IRVerificationError(f"instruction {instruction!r} is missing an opcode")
    return opcode
