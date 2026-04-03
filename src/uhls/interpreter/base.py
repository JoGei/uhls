"""Duck-typed shared execution engine for canonical µhLS IRs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from uhls.middleend.uir import ArrayType, IncomingValue, Literal, Parameter, Variable, normalize_type

from .eval import eval_binary, eval_compare, eval_unary, normalize_int, truthy
from .runtime import ExecutionResult, ExecutionState, InterpreterError

_MISSING = object()
_UNSET = object()
_TERMINATORS = {"br", "cbr", "ret"}
_UNARY_OPS = {"mov", "neg", "not"}
_BINARY_OPS = {"add", "sub", "mul", "div", "mod", "and", "or", "xor", "shl", "shr"}
_COMPARE_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}


def _lookup(obj: Any, *names: str, default: Any = _UNSET) -> Any:
    """Try several attribute/key spellings on a duck-typed IR object.

    Args:
        obj: Mapping- or object-like IR node to inspect.
        *names: Candidate key or attribute names in priority order.
        default: Value returned when none of the names exist.

    Returns:
        The first matching value, or ``default`` when provided.
    """
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not _UNSET:
        return default
    raise AttributeError(f"missing any of attributes/keys {names!r} on {obj!r}")


def _sequence(value: Any) -> list[Any]:
    """Convert a sequence-like IR field into a plain list.

    Args:
        value: Sequence, tuple, mapping, iterable, or ``None``.

    Returns:
        A list view suitable for uniform interpreter logic.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    if isinstance(value, Mapping):
        return list(value.values())
    return list(value)


def _value_name(value: Any) -> str:
    """Extract a symbolic name from a scalar-like IR object.

    Args:
        value: String or object carrying a name/id-style attribute.

    Returns:
        The resolved symbol name.
    """
    if isinstance(value, (Parameter, Variable)):
        return value.name
    if isinstance(value, str):
        return value
    named = _lookup(value, "name", "id", "identifier", "symbol", default=_MISSING)
    if named is not _MISSING:
        return str(named)
    raise InterpreterError(f"cannot extract scalar name from {value!r}")


def _value_type(value: Any) -> Any | None:
    """Extract type metadata from a value-like IR object."""
    if isinstance(value, (Parameter, Variable, Literal)):
        return value.type
    return _lookup(value, "type", "ty", "typ", "dtype", default=None)


def _value_literal(value: Any) -> tuple[bool, int, Any | None]:
    """Recognize integer literal operands across flexible IR shapes.

    Args:
        value: Candidate operand object, integer, or literal wrapper.

    Returns:
        A tuple ``(is_literal, literal_value, literal_type)``.
    """
    if isinstance(value, Literal):
        return True, value.value, value.type
    if isinstance(value, bool):
        return True, int(value), "i1"
    if isinstance(value, int):
        return True, int(value), None

    literal = _lookup(value, "value", "literal", "const", default=_MISSING)
    if literal is _MISSING:
        return False, 0, None

    name = _lookup(value, "name", "id", "identifier", "symbol", default=_MISSING)
    if name is not _MISSING:
        kind = _lookup(value, "kind", "tag", "role", default=None)
        if kind not in {"const", "literal", "immediate"}:
            return False, 0, None

    return True, int(literal), _value_type(value)


@dataclass(frozen=True)
class TerminatorAction:
    kind: str
    target: str | None = None
    value: int | None = None


class BaseAdapter:
    """Common duck-typed accessors for canonical µhLS IR objects."""

    def function_name(self, function: Any) -> str:
        """Return the function's human-readable name."""
        return str(_lookup(function, "name", "id", default="<anon>"))

    def function_return_type(self, function: Any) -> Any | None:
        """Return the function's declared return type metadata if present."""
        return _lookup(function, "return_type", "ret_type", "type", default=None)

    def parameters(self, function: Any) -> list[Any]:
        """Return the function parameter list as a plain Python list."""
        return _sequence(_lookup(function, "params", "parameters", "arguments", default=[]))

    def parameter_name(self, parameter: Any) -> str:
        """Return a parameter's symbolic name."""
        return _value_name(parameter)

    def parameter_type(self, parameter: Any) -> Any | None:
        """Return a parameter's declared type metadata."""
        return _value_type(parameter)

    def call_callee(self, instruction: Any) -> str:
        """Return the direct callee symbol used by ``instruction``."""
        return self._as_symbol_name(_lookup(instruction, "callee", "function", "target"))

    def call_operands(self, instruction: Any) -> list[Any]:
        """Return explicit call operands for canonical call instructions."""
        return _sequence(_lookup(instruction, "operands", "args", "arguments", "values", default=[]))

    def call_arg_count(self, instruction: Any) -> int | None:
        """Return the lowered call argument count when present."""
        raw = _lookup(instruction, "arg_count", "argc", "count", default=None)
        return None if raw is None else int(raw)

    def param_index(self, instruction: Any) -> int:
        """Return the positional slot carried by one ``param`` pseudo-op."""
        return int(_lookup(instruction, "index", "slot", "position"))

    def param_value(self, instruction: Any) -> Any:
        """Return the operand carried by one ``param`` pseudo-op."""
        raw = _lookup(instruction, "value", "operand", "arg", "src", default=_MISSING)
        if raw is not _MISSING:
            return raw
        operands = self.operands(instruction)
        if len(operands) >= 2:
            return operands[1]
        raise InterpreterError(f"cannot determine param value for {instruction!r}")

    def blocks(self, function: Any) -> dict[str, Any]:
        """Return the function's basic blocks indexed by label.

        Args:
            function: Duck-typed function object exposing a block collection.
        """
        raw_blocks = _lookup(function, "blocks", "basic_blocks", "body", default=None)
        if raw_blocks is None:
            raise InterpreterError(f"function '{self.function_name(function)}' has no blocks")

        if isinstance(raw_blocks, Mapping):
            block_map = {str(name): block for name, block in raw_blocks.items()}
        else:
            block_map = {}
            for block in _sequence(raw_blocks):
                label = self.block_label(block)
                block_map[label] = block

        if not block_map:
            raise InterpreterError(f"function '{self.function_name(function)}' has no basic blocks")
        return block_map

    def entry_label(self, function: Any, blocks: Mapping[str, Any]) -> str:
        """Determine the entry block label for ``function``.

        Args:
            function: Function being executed.
            blocks: Already-materialized block map for that function.
        """
        explicit = _lookup(function, "entry", "entry_block", "entry_label", default=_MISSING)
        if explicit is not _MISSING:
            if isinstance(explicit, str):
                return explicit
            return self.block_label(explicit)
        return next(iter(blocks))

    def block_label(self, block: Any) -> str:
        """Return a block's label/name."""
        return str(_lookup(block, "label", "name", "id"))

    def split_block(self, block: Any) -> tuple[list[Any], Any]:
        """Split a block into non-terminators and its final terminator.

        Args:
            block: Duck-typed basic block object.
        """
        instructions = _sequence(_lookup(block, "instructions", "ops", "body", default=[]))
        terminator = _lookup(block, "terminator", "term", "exit", default=None)
        if terminator is not None:
            return instructions, terminator

        if not instructions:
            raise InterpreterError(f"block '{self.block_label(block)}' has no terminator")

        terminator = instructions[-1]
        if self.opcode(terminator) not in _TERMINATORS:
            raise InterpreterError(f"block '{self.block_label(block)}' does not end in a terminator")
        return instructions[:-1], terminator

    def opcode(self, instruction: Any) -> str:
        """Return an instruction opcode normalized to lowercase text."""
        return str(_lookup(instruction, "opcode", "op", "mnemonic", "kind")).lower()

    def result_name(self, instruction: Any) -> str:
        """Return the destination/result symbol defined by ``instruction``."""
        return _value_name(_lookup(instruction, "dest", "result", "name"))

    def result_type(self, instruction: Any) -> Any | None:
        """Return the declared result type for ``instruction``."""
        return _value_type(instruction)

    def operands(self, instruction: Any) -> list[Any]:
        """Return an instruction's operands in execution order.

        Args:
            instruction: Duck-typed instruction node with explicit or implicit
                operand fields.
        """
        raw = _lookup(instruction, "operands", "args", "inputs", "values", default=_MISSING)
        if raw is not _MISSING:
            return _sequence(raw)

        if self.opcode(instruction) in {
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
            left = _lookup(instruction, "lhs", "left")
            right = _lookup(instruction, "rhs", "right")
            return [left, right]

        if self.opcode(instruction) in {"mov", "neg", "not"}:
            return [_lookup(instruction, "value", "operand", "arg", "src", default=None)]

        return []

    def array_name(self, instruction: Any) -> str:
        """Return the symbolic array operand used by a memory instruction."""
        raw = _lookup(instruction, "array", "mem", "base", default=_MISSING)
        if raw is not _MISSING:
            return self._as_symbol_name(raw)

        operands = self.operands(instruction)
        if operands:
            return self._as_symbol_name(operands[0])
        raise InterpreterError(f"cannot determine array operand for {instruction!r}")

    def index_operand(self, instruction: Any) -> Any:
        """Return the index operand used by a ``load`` or ``store`` instruction."""
        raw = _lookup(instruction, "index", "idx", default=_MISSING)
        if raw is not _MISSING:
            return raw

        operands = self.operands(instruction)
        if len(operands) >= 2:
            return operands[1]
        raise InterpreterError(f"cannot determine index operand for {instruction!r}")

    def store_value_operand(self, instruction: Any) -> Any:
        """Return the value operand consumed by a ``store`` instruction."""
        raw = _lookup(instruction, "value", "src", "source", default=_MISSING)
        if raw is not _MISSING:
            return raw

        operands = self.operands(instruction)
        if len(operands) >= 3:
            return operands[2]
        raise InterpreterError(f"cannot determine store value for {instruction!r}")

    def branch_target(self, instruction: Any) -> str:
        """Return the destination block label for an unconditional branch."""
        raw = _lookup(instruction, "target", "label", "dest", default=_MISSING)
        if raw is not _MISSING:
            return self._as_symbol_name(raw)

        operands = self.operands(instruction)
        if operands:
            return self._as_symbol_name(operands[0])
        raise InterpreterError(f"cannot determine branch target for {instruction!r}")

    def branch_condition(self, instruction: Any) -> Any:
        """Return the condition operand for a conditional branch."""
        raw = _lookup(instruction, "cond", "condition", default=_MISSING)
        if raw is not _MISSING:
            return raw

        operands = self.operands(instruction)
        if operands:
            return operands[0]
        raise InterpreterError(f"cannot determine branch condition for {instruction!r}")

    def cbr_targets(self, instruction: Any) -> tuple[str, str]:
        """Return the true and false targets for a conditional branch."""
        true_target = _lookup(
            instruction,
            "true_target",
            "then_target",
            "if_true",
            "true_block",
            "then_block",
            default=_MISSING,
        )
        false_target = _lookup(
            instruction,
            "false_target",
            "else_target",
            "if_false",
            "false_block",
            "else_block",
            default=_MISSING,
        )

        operands = self.operands(instruction)
        if true_target is _MISSING and len(operands) >= 2:
            true_target = operands[1]
        if false_target is _MISSING and len(operands) >= 3:
            false_target = operands[2]

        if true_target is _MISSING or false_target is _MISSING:
            raise InterpreterError(f"cannot determine cbr targets for {instruction!r}")

        return self._as_symbol_name(true_target), self._as_symbol_name(false_target)

    def return_operand(self, instruction: Any) -> Any | None:
        """Return the optional operand of a ``ret`` terminator."""
        raw = _lookup(instruction, "value", "operand", "arg", default=_MISSING)
        if raw is not _MISSING:
            return raw

        operands = self.operands(instruction)
        return operands[0] if operands else None

    def phi_incomings(self, instruction: Any) -> list[tuple[str, Any]]:
        """Return ``(predecessor, operand)`` pairs for one ``phi`` instruction."""
        incoming = _lookup(instruction, "incoming", "inputs", "sources", "pairs", default=_MISSING)
        if incoming is _MISSING:
            raise InterpreterError(f"phi instruction {instruction!r} has no incoming values")

        result: list[tuple[str, Any]] = []
        if isinstance(incoming, Mapping):
            for pred, value in incoming.items():
                result.append((self._as_symbol_name(pred), value))
            return result

        for item in _sequence(incoming):
            if isinstance(item, IncomingValue):
                result.append((item.pred, item.value))
                continue
            if isinstance(item, tuple) and len(item) == 2:
                pred, value = item
                result.append((self._as_symbol_name(pred), value))
                continue

            pred = _lookup(item, "pred", "predecessor", "block", "label", default=_MISSING)
            value = _lookup(item, "value", "operand", "arg", "src", default=_MISSING)
            if pred is _MISSING or value is _MISSING:
                raise InterpreterError(f"unsupported phi incoming shape: {item!r}")
            result.append((self._as_symbol_name(pred), value))

        return result

    def print_format(self, instruction: Any) -> str:
        """Return the format string carried by one print instruction."""
        return str(_lookup(instruction, "format", "fmt", "text"))

    def print_operands(self, instruction: Any) -> list[Any]:
        """Return the scalar operands carried by one print instruction."""
        return _sequence(_lookup(instruction, "operands", "args", "arguments", "values", default=[]))

    def on_block_entry(
        self,
        interpreter: "InterpreterBase",
        function: Any,
        block: Any,
        state: ExecutionState,
        predecessor: str | None,
    ) -> None:
        """Hook for block-entry actions such as SSA phi resolution.

        Args:
            interpreter: Active interpreter instance.
            function: Function being executed.
            block: Block that is about to execute.
            state: Mutable execution state.
            predecessor: Previously executed block label, or ``None`` on entry.
        """
        return None

    def body_instructions(self, block: Any) -> list[Any]:
        """Return the block instructions that should execute before the terminator."""
        body, _ = self.split_block(block)
        return body

    def terminator(self, block: Any) -> Any:
        """Return the block's terminating control-flow instruction."""
        _, terminator = self.split_block(block)
        return terminator

    def _as_symbol_name(self, value: Any) -> str:
        """Extract a label/name from a block-like or symbol-like object."""
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for name in ("label", "name", "id", "identifier", "symbol", "target"):
                if name in value:
                    return str(value[name])
        for name in ("label", "name", "id", "identifier", "symbol", "target"):
            if hasattr(value, name):
                return str(getattr(value, name))
        raise InterpreterError(f"cannot extract label/name from {value!r}")


class InterpreterBase:
    """Shared explicit-CFG interpreter with adapter-driven IR details."""

    def __init__(self, adapter: BaseAdapter | None = None) -> None:
        """Create an interpreter bound to a specific IR adapter.

        Args:
            adapter: Accessor object that knows how to read the concrete IR
                node shapes. Defaults to :class:`BaseAdapter`.
        """
        self.adapter = adapter or BaseAdapter()

    def run(
        self,
        function: Any,
        arguments: Mapping[str, Any] | Sequence[Any] | None = None,
        arrays: Mapping[str, Any] | None = None,
        module: Any | None = None,
        *,
        trace: bool = False,
        step_limit: int = 100_000,
        state: ExecutionState | None = None,
    ) -> ExecutionResult:
        """Execute one function until an explicit ``ret`` is reached.

        Args:
            function: Duck-typed function object containing parameters and blocks.
            arguments: Either a name->value mapping or positional argument list.
            arrays: Optional array-memory initializers.
            module: Optional module-like object used to resolve direct calls.
            trace: Whether to append human-readable trace events to the state.
            step_limit: Maximum executed instruction/terminator count before aborting.
            state: Optional pre-existing execution state to reuse.

        Returns:
            An :class:`ExecutionResult` containing the return value and final state.
        """
        execution_state = state or ExecutionState(trace_enabled=trace)
        execution_state.trace_enabled = trace or execution_state.trace_enabled
        execution_state.pending_params.clear()
        if arrays:
            execution_state.memory.initialize(arrays)
        local_arrays = getattr(function, "local_arrays", None)
        if local_arrays:
            execution_state.memory.initialize(local_arrays)

        self._bind_arguments(function, arguments, execution_state)

        blocks = self.adapter.blocks(function)
        current_label = self.adapter.entry_label(function, blocks)
        predecessor = None

        while True:
            block = blocks.get(current_label)
            if block is None:
                raise InterpreterError(f"jump to unknown block '{current_label}'")

            execution_state.current_block = current_label
            execution_state.predecessor_block = predecessor
            detail = f"enter {current_label}"
            if predecessor is not None:
                detail = f"{detail} from {predecessor}"
            execution_state.record("block", block=current_label, detail=detail)

            self.adapter.on_block_entry(self, function, block, execution_state, predecessor)

            for instruction in self.adapter.body_instructions(block):
                self._tick(execution_state, step_limit)
                self._execute_instruction(function, module, block, instruction, execution_state, step_limit)

            terminator = self.adapter.terminator(block)
            self._tick(execution_state, step_limit)
            action = self._execute_terminator(function, block, terminator, execution_state)

            if action.kind == "jump":
                predecessor = current_label
                current_label = action.target or ""
                continue

            if action.kind == "return":
                execution_state.returned = True
                execution_state.return_value = action.value
                return ExecutionResult(return_value=action.value, state=execution_state)

            raise InterpreterError(f"unknown terminator action '{action.kind}'")

    def _bind_arguments(
        self,
        function: Any,
        arguments: Mapping[str, Any] | Sequence[Any] | None,
        state: ExecutionState,
    ) -> None:
        """Normalize and bind function arguments into ``state.env``.

        Args:
            function: Function whose parameters define the expected inputs.
            arguments: Mapping or positional argument sequence supplied by the caller.
            state: Execution state receiving the bound argument values.
        """
        parameters = self.adapter.parameters(function)
        if not parameters:
            return

        scalar_parameters = [
            parameter
            for parameter in parameters
            if not isinstance(normalize_type(self.adapter.parameter_type(parameter)), ArrayType)
        ]
        if not scalar_parameters:
            return

        if arguments is None:
            raise InterpreterError(
                f"function '{self.adapter.function_name(function)}' requires {len(scalar_parameters)} arguments"
            )

        if isinstance(arguments, Mapping):
            for parameter in scalar_parameters:
                name = self.adapter.parameter_name(parameter)
                if name not in arguments:
                    raise InterpreterError(f"missing argument '{name}'")
                type_hint = self.adapter.parameter_type(parameter)
                value = normalize_int(int(arguments[name]), type_hint)
                state.bind_scalar(name, value, type_hint)
            return

        values = list(arguments)
        if len(values) != len(scalar_parameters):
            raise InterpreterError(
                f"function '{self.adapter.function_name(function)}' expected {len(scalar_parameters)} arguments, got {len(values)}"
            )

        for parameter, raw_value in zip(scalar_parameters, values, strict=True):
            name = self.adapter.parameter_name(parameter)
            type_hint = self.adapter.parameter_type(parameter)
            state.bind_scalar(name, normalize_int(int(raw_value), type_hint), type_hint)

    def _tick(self, state: ExecutionState, step_limit: int) -> None:
        """Advance the step counter and enforce the configured limit."""
        state.steps += 1
        if state.steps > step_limit:
            raise InterpreterError(f"step limit exceeded ({step_limit})")

    def _execute_instruction(
        self,
        function: Any,
        module: Any | None,
        block: Any,
        instruction: Any,
        state: ExecutionState,
        step_limit: int,
    ) -> None:
        """Execute one non-terminator instruction.

        Args:
            function: Enclosing function, used for type context when needed.
            module: Optional module-like object used for direct-call lookup.
            block: Enclosing basic block for trace messages.
            instruction: Instruction to execute.
            state: Mutable execution state.
            step_limit: Maximum instruction budget shared across nested calls.
        """
        opcode = self.adapter.opcode(instruction)
        block_label = self.adapter.block_label(block)

        if opcode == "phi":
            raise InterpreterError("phi must be handled on block entry")
        if state.pending_params and opcode not in {"param", "call"}:
            raise InterpreterError("param ops must appear immediately before a call")

        if opcode == "const":
            raw_value = _lookup(instruction, "value", "literal", "const", default=_MISSING)
            if raw_value is _MISSING:
                operands = self.adapter.operands(instruction)
                if not operands:
                    raise InterpreterError(f"const instruction missing literal: {instruction!r}")
                _, raw_value, literal_type = _value_literal(operands[0])
                type_hint = self.adapter.result_type(instruction) or literal_type
            else:
                type_hint = self.adapter.result_type(instruction)
            result = normalize_int(int(raw_value), type_hint)
            self._assign(instruction, result, state)
            state.record(
                "exec",
                block=block_label,
                opcode=opcode,
                detail=f"{self.adapter.result_name(instruction)} = {result}",
            )
            return

        if opcode in _UNARY_OPS:
            operand = self._resolve_operand(self.adapter.operands(instruction)[0], state)
            result = eval_unary(opcode, operand, self.adapter.result_type(instruction))
            self._assign(instruction, result, state)
            state.record(
                "exec",
                block=block_label,
                opcode=opcode,
                detail=f"{self.adapter.result_name(instruction)} = {result}",
            )
            return

        if opcode in _BINARY_OPS:
            left, right = self.adapter.operands(instruction)
            lhs = self._resolve_operand(left, state)
            rhs = self._resolve_operand(right, state)
            result = eval_binary(opcode, lhs, rhs, self.adapter.result_type(instruction))
            self._assign(instruction, result, state)
            state.record(
                "exec",
                block=block_label,
                opcode=opcode,
                detail=f"{self.adapter.result_name(instruction)} = {result}",
            )
            return

        if opcode in _COMPARE_OPS:
            left, right = self.adapter.operands(instruction)
            lhs = self._resolve_operand(left, state)
            rhs = self._resolve_operand(right, state)
            result = eval_compare(opcode, lhs, rhs)
            self._assign(instruction, result, state)
            state.record(
                "exec",
                block=block_label,
                opcode=opcode,
                detail=f"{self.adapter.result_name(instruction)} = {result}",
            )
            return

        if opcode == "load":
            array_name = self.adapter.array_name(instruction)
            index = self._resolve_operand(self.adapter.index_operand(instruction), state)
            raw_value = state.memory.load(array_name, index)
            type_hint = self.adapter.result_type(instruction) or state.memory.element_type(array_name)
            result = normalize_int(raw_value, type_hint)
            self._assign(instruction, result, state)
            state.record(
                "exec",
                block=block_label,
                opcode=opcode,
                detail=f"{self.adapter.result_name(instruction)} = {array_name}[{int(index)}] -> {result}",
            )
            return

        if opcode == "store":
            array_name = self.adapter.array_name(instruction)
            index = self._resolve_operand(self.adapter.index_operand(instruction), state)
            value = self._resolve_operand(self.adapter.store_value_operand(instruction), state)
            element_type = state.memory.element_type(array_name)
            stored = normalize_int(value, element_type)
            state.memory.store(array_name, index, stored)
            state.record(
                "exec",
                block=block_label,
                opcode=opcode,
                detail=f"{array_name}[{int(index)}] = {stored}",
            )
            return

        if opcode == "param":
            index = self.adapter.param_index(instruction)
            state.pending_params.append((index, self.adapter.param_value(instruction)))
            state.record(
                "exec",
                block=block_label,
                opcode=opcode,
                detail=f"param {index}",
            )
            return

        if opcode == "call":
            result = self._execute_call(function, module, block_label, instruction, state, step_limit)
            if getattr(instruction, "dest", None) is not None:
                self._assign(instruction, result, state)
            state.record(
                "exec",
                block=block_label,
                opcode=opcode,
                detail=(
                    f"{self.adapter.result_name(instruction)} = {self.adapter.call_callee(instruction)}(...) -> {result}"
                    if getattr(instruction, "dest", None) is not None
                    else f"{self.adapter.call_callee(instruction)}(...)"
                ),
            )
            return

        if opcode == "print":
            rendered = self._format_print(
                self.adapter.print_format(instruction),
                [self._resolve_operand(operand, state) for operand in self.adapter.print_operands(instruction)],
            )
            state.stdout.append(rendered)
            state.record(
                "exec",
                block=block_label,
                opcode=opcode,
                detail=rendered,
            )
            return

        raise InterpreterError(f"unsupported opcode '{opcode}'")

    def _execute_terminator(
        self,
        function: Any,
        block: Any,
        instruction: Any,
        state: ExecutionState,
    ) -> TerminatorAction:
        """Execute one terminator and describe the resulting control-flow action.

        Args:
            function: Enclosing function, used for return-type normalization.
            block: Current basic block.
            instruction: Terminator instruction to execute.
            state: Mutable execution state.
        """
        opcode = self.adapter.opcode(instruction)
        block_label = self.adapter.block_label(block)

        if state.pending_params:
            raise InterpreterError("param ops must appear immediately before a call")

        if opcode == "br":
            target = self.adapter.branch_target(instruction)
            state.record("branch", block=block_label, opcode=opcode, detail=f"-> {target}")
            return TerminatorAction(kind="jump", target=target)

        if opcode == "cbr":
            condition = self._resolve_operand(self.adapter.branch_condition(instruction), state)
            true_target, false_target = self.adapter.cbr_targets(instruction)
            target = true_target if truthy(condition) else false_target
            state.record(
                "branch",
                block=block_label,
                opcode=opcode,
                detail=f"{condition} -> {target}",
            )
            return TerminatorAction(kind="jump", target=target)

        if opcode == "ret":
            operand = self.adapter.return_operand(instruction)
            value = None if operand is None else self._resolve_operand(operand, state)
            value = normalize_int(value, self.adapter.function_return_type(function)) if value is not None else None
            state.record("return", block=block_label, opcode=opcode, detail=f"return {value}")
            return TerminatorAction(kind="return", value=value)

        raise InterpreterError(f"unsupported terminator opcode '{opcode}'")

    def _assign(self, instruction: Any, value: int, state: ExecutionState) -> None:
        """Write an instruction result into the scalar environment."""
        state.bind_scalar(
            self.adapter.result_name(instruction),
            value,
            self.adapter.result_type(instruction),
        )

    def _execute_call(
        self,
        function: Any,
        module: Any | None,
        block_label: str,
        instruction: Any,
        state: ExecutionState,
        step_limit: int,
    ) -> int | None:
        """Execute one direct call and return its optional scalar result."""
        callee_name = self.adapter.call_callee(instruction)
        callee = self._lookup_callee(module, callee_name)
        operands = self.adapter.call_operands(instruction)
        arg_count = self.adapter.call_arg_count(instruction)

        if arg_count is not None:
            if operands:
                raise InterpreterError(f"lowered call '{callee_name}' cannot also carry explicit operands")
            operands = self._consume_pending_params(state, arg_count)
        elif state.pending_params:
            raise InterpreterError(f"direct call '{callee_name}' cannot consume pending param ops")

        scalar_arguments, array_arguments = self._prepare_call_arguments(callee, operands, state)
        child_state = ExecutionState(
            memory=state.memory,
            trace_enabled=state.trace_enabled,
            trace=state.trace,
            stdout=state.stdout,
            steps=state.steps,
        )
        child_result = self.run(
            callee,
            scalar_arguments,
            arrays=array_arguments,
            module=module,
            trace=state.trace_enabled,
            step_limit=step_limit,
            state=child_state,
        )
        state.steps = child_state.steps
        state.record(
            "call",
            block=block_label,
            opcode="call",
            detail=f"{self.adapter.function_name(function)} -> {callee_name} returned {child_result.return_value}",
        )
        return child_result.return_value

    def _lookup_callee(self, module: Any | None, name: str) -> Any:
        """Resolve one direct callee from a module-like container."""
        if module is None:
            raise InterpreterError(f"call to '{name}' requires module context")

        callee = None
        getter = getattr(module, "get_function", None)
        if callable(getter):
            callee = getter(name)
        elif isinstance(module, Mapping):
            callee = module.get(name)
        else:
            function_map = getattr(module, "function_map", None)
            if callable(function_map):
                callee = function_map().get(name)

        if callee is None:
            raise InterpreterError(f"call to unknown function '{name}'")
        return callee

    def _consume_pending_params(self, state: ExecutionState, arg_count: int) -> list[Any]:
        """Consume one lowered call's pending ``param`` operands in order."""
        if arg_count < 0:
            raise InterpreterError(f"call argument count must be non-negative, got {arg_count}")
        if len(state.pending_params) != arg_count:
            raise InterpreterError(f"call expected {arg_count} pending param ops, got {len(state.pending_params)}")

        ordered = sorted(state.pending_params, key=lambda item: item[0])
        expected_zero = list(range(arg_count))
        expected_one = list(range(1, arg_count + 1))
        indices = [index for index, _ in ordered]
        if indices != expected_zero and indices != expected_one:
            raise InterpreterError(
                f"param indices must be contiguous and start at 0 or 1, got {', '.join(str(index) for index in indices)}"
            )
        state.pending_params.clear()
        return [operand for _, operand in ordered]

    def _prepare_call_arguments(
        self,
        callee: Any,
        operands: Sequence[Any],
        state: ExecutionState,
    ) -> tuple[dict[str, int], dict[str, dict[str, object]]]:
        """Resolve one call site's operands into scalar and array arguments."""
        parameters = self.adapter.parameters(callee)
        if len(operands) != len(parameters):
            raise InterpreterError(
                f"call to '{self.adapter.function_name(callee)}' expected {len(parameters)} arguments, got {len(operands)}"
            )

        scalar_arguments: dict[str, int] = {}
        array_arguments: dict[str, dict[str, object]] = {}
        for parameter, operand in zip(parameters, operands, strict=True):
            parameter_name = self.adapter.parameter_name(parameter)
            parameter_type = normalize_type(self.adapter.parameter_type(parameter))
            if isinstance(parameter_type, ArrayType):
                array_name = self._resolve_array_operand_name(operand)
                if not state.memory.has(array_name):
                    raise InterpreterError(
                        f"call argument '{array_name}' for array parameter '{parameter_name}' is not bound in memory"
                    )
                array_arguments[parameter_name] = {"alias": array_name}
                continue
            scalar_arguments[parameter_name] = self._resolve_operand(operand, state)
        return scalar_arguments, array_arguments

    def _resolve_array_operand_name(self, operand: Any) -> str:
        """Resolve one array-valued operand to its symbolic memory name."""
        if isinstance(operand, (Parameter, Variable)):
            return operand.name
        if isinstance(operand, str):
            return operand

        name = _lookup(operand, "name", "id", "identifier", "symbol", default=_MISSING)
        if name is not _MISSING:
            return str(name)
        raise InterpreterError(f"array arguments must be passed by symbolic name, got {operand!r}")

    def _resolve_operand(self, operand: Any, state: ExecutionState) -> int:
        """Resolve a literal or named operand to an integer value.

        Args:
            operand: IR operand object, literal, or scalar name.
            state: Execution state providing the current scalar environment.
        """
        is_literal, literal, literal_type = _value_literal(operand)
        if is_literal:
            return normalize_int(literal, literal_type)

        if isinstance(operand, (Parameter, Variable)):
            return state.read_scalar(operand.name)

        if isinstance(operand, str):
            return state.read_scalar(operand)

        name = _lookup(operand, "name", "id", "identifier", "symbol", default=_MISSING)
        if name is not _MISSING:
            return state.read_scalar(str(name))

        raise InterpreterError(f"cannot resolve operand {operand!r}")

    def _format_print(self, format_text: str, values: Sequence[int]) -> str:
        """Render one print operation using a small printf-style subset."""
        if not values:
            try:
                return format_text % ()
            except TypeError:
                return format_text
        try:
            return format_text % tuple(values)
        except (TypeError, ValueError) as exc:
            raise InterpreterError(f"invalid print format {format_text!r}") from exc
