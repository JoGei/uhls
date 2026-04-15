"""Interpreter for seq-style µhIR designs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

from uhls.backend.hls.uhir import UHIRDesign, UHIRNode, UHIRRegion
from uhls.backend.hls.uhir.gopt import project_to_seq_design
from uhls.backend.hls.uhir.gopt.loops import collect_explicit_loops, collect_loop_candidates
from uhls.backend.hls.uhir.pretty import _order_region_nodes
from uhls.interpreter.eval import eval_binary, eval_compare, eval_unary, normalize_int, truthy
from uhls.middleend.uir import ArrayType, normalize_type

from .runtime import ExecutionResult, ExecutionState, InterpreterError

_UNARY_OPS = {"mov", "neg", "not"}
_BINARY_OPS = {"add", "sub", "mul", "div", "mod", "and", "or", "xor", "shl", "shr"}
_COMPARE_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}
_MEMREF_RE = re.compile(r"^memref<\s*([^,\s>]+)\s*(?:,\s*([0-9]+)\s*)?>$")
_TYPED_LITERAL_RE = re.compile(r"^([+-]?[0-9]+):([A-Za-z][A-Za-z0-9_]*)$")


@dataclass(frozen=True)
class _RegionOutcome:
    returned: bool = False
    return_value: int | None = None


@dataclass(frozen=True)
class _ProcedureParam:
    name: str
    kind: str
    type_hint: Any | None = None


@dataclass(frozen=True)
class _LoopBranchInfo:
    region_id: str
    branch_id: str
    header_start_index: int
    body_child_id: str
    exit_child_id: str
    backedge_label: str | None
    exit_label: str | None


class UHIRInterpreter:
    """Execute one seq-stage µhIR design directly."""

    def __init__(self) -> None:
        self.design: UHIRDesign | None = None
        self.region_by_id: dict[str, UHIRRegion] = {}
        self.node_by_region: dict[str, dict[str, UHIRNode]] = {}
        self.ordered_nodes_by_region: dict[str, list[UHIRNode]] = {}
        self.region_entry_label: dict[str, str] = {}
        self.loop_branches: dict[tuple[str, str], _LoopBranchInfo] = {}
        self.node_mappings: dict[tuple[str, str], tuple[str, ...]] = {}
        self.procedure_params: dict[str, tuple[_ProcedureParam, ...]] = {}
        self.procedure_return_types: dict[str, Any | None] = {}
        self.global_scalars: dict[str, int] = {}
        self.global_scalar_types: dict[str, Any] = {}
        self.active_static_loop_counters: dict[tuple[str, str], int] = {}

    def run(
        self,
        design: UHIRDesign,
        arguments: dict[str, int] | None = None,
        arrays: dict[str, dict[str, object]] | None = None,
        *,
        function_name: str | None = None,
        trace: bool = False,
        step_limit: int = 100_000,
        state: ExecutionState | None = None,
    ) -> ExecutionResult:
        """Execute one seq-style µhIR design."""
        active_design = design if design.stage == "seq" else project_to_seq_design(design)
        if active_design.stage != "seq":
            raise InterpreterError(f"µhIR interpreter expects seq-style input, got stage '{design.stage}'")

        self._prepare(active_design)

        top_region_id = self._select_entry_procedure(function_name)
        execution_state = state or ExecutionState(trace_enabled=trace)
        execution_state.trace_enabled = trace or execution_state.trace_enabled
        execution_state.pending_params.clear()
        execution_state.returned = False
        execution_state.return_value = None
        execution_state.current_block = None
        execution_state.predecessor_block = None
        self.active_static_loop_counters = {}

        execution_state.memory.initialize(arrays)
        self._push_frame(execution_state, top_region_id, arguments or {}, {})
        try:
            outcome = self._execute_region(
                self.region_by_id[top_region_id],
                execution_state,
                step_limit=step_limit,
                predecessor_label=self.region_entry_label.get(top_region_id),
            )
        finally:
            self._pop_frame(execution_state)

        execution_state.returned = outcome.returned
        execution_state.return_value = outcome.return_value
        return ExecutionResult(return_value=outcome.return_value, state=execution_state)

    def _prepare(self, design: UHIRDesign) -> None:
        self.design = design
        self.region_by_id = {region.id: region for region in design.regions}
        self.node_by_region = {
            region.id: {node.id: node for node in region.nodes}
            for region in design.regions
        }
        self.ordered_nodes_by_region = {
            region.id: list(_order_region_nodes(region))
            for region in design.regions
        }
        self.node_mappings = {}
        for region in design.regions:
            for node in region.nodes:
                names = [node.id]
                names.extend(mapping.source_id for mapping in region.mappings if mapping.node_id == node.id)
                self.node_mappings[(region.id, node.id)] = tuple(dict.fromkeys(names))

        self.global_scalars = {}
        self.global_scalar_types = {}
        for const_decl in design.constants:
            type_hint = normalize_type(const_decl.type)
            if isinstance(const_decl.value, int):
                value = normalize_int(const_decl.value, type_hint)
            else:
                value = self._parse_literal_int(str(const_decl.value))
                value = normalize_int(value, type_hint)
            self.global_scalars[const_decl.name] = value
            if type_hint is not None:
                self.global_scalar_types[const_decl.name] = type_hint

        self.procedure_params = {}
        self.procedure_return_types = {}
        for region in design.regions:
            if region.kind != "procedure":
                continue
            if self._is_entry_procedure(region.id):
                params = []
                for port in design.inputs:
                    kind, type_hint = self._port_param_kind(port.type)
                    params.append(_ProcedureParam(port.name, kind, type_hint))
                self.procedure_params[region.id] = tuple(params)
                output_port = next((port for port in design.outputs if port.name == "result"), None)
                self.procedure_return_types[region.id] = None if output_port is None else normalize_type(output_port.type)
                continue
            self.procedure_params[region.id] = self._infer_procedure_params(region)
            self.procedure_return_types[region.id] = None

        self.region_entry_label = {}
        self.loop_branches = {}
        for candidate in collect_loop_candidates(design):
            ordered_nodes = self.ordered_nodes_by_region[candidate.parent_region.id]
            header_nodes = [node for node in ordered_nodes if node.id in candidate.header_node_ids]
            header_start_index = min(ordered_nodes.index(node) for node in header_nodes)
            info = _LoopBranchInfo(
                region_id=candidate.parent_region.id,
                branch_id=candidate.branch_node.id,
                header_start_index=header_start_index,
                body_child_id=candidate.body_region.id,
                exit_child_id=candidate.empty_region.id,
                backedge_label=self._text_attr(candidate.branch_node, "true_input_label"),
                exit_label=self._text_attr(candidate.branch_node, "false_input_label"),
            )
            self.loop_branches[(info.region_id, info.branch_id)] = info
            entry_label = self._infer_loop_entry_label(
                candidate.parent_region,
                candidate.header_node_ids,
                excluded_labels={info.backedge_label} if info.backedge_label is not None else set(),
            )
            if entry_label is not None and candidate.parent_region.id not in self.region_entry_label:
                self.region_entry_label[candidate.parent_region.id] = entry_label

        for explicit in collect_explicit_loops(design):
            if explicit.header_branch is None or explicit.body_region is None or explicit.empty_region is None:
                continue
            ordered_nodes = self.ordered_nodes_by_region[explicit.header_region.id]
            header_nodes = [
                node
                for node in ordered_nodes
                if not (node.opcode == "nop" and node.attributes.get("role") in {"source", "sink"})
            ]
            if not header_nodes:
                continue
            header_start_index = min(ordered_nodes.index(node) for node in header_nodes)
            info = _LoopBranchInfo(
                region_id=explicit.header_region.id,
                branch_id=explicit.header_branch.id,
                header_start_index=header_start_index,
                body_child_id=explicit.body_region.id,
                exit_child_id=explicit.empty_region.id,
                backedge_label=self._text_attr(explicit.header_branch, "true_input_label"),
                exit_label=self._text_attr(explicit.header_branch, "false_input_label"),
            )
            self.loop_branches[(info.region_id, info.branch_id)] = info
            entry_label = self._infer_loop_entry_label(
                explicit.header_region,
                {node.id for node in header_nodes},
                excluded_labels={info.backedge_label} if info.backedge_label is not None else set(),
            )
            if entry_label is not None and explicit.header_region.id not in self.region_entry_label:
                self.region_entry_label[explicit.header_region.id] = entry_label

    def _select_entry_procedure(self, requested: str | None) -> str:
        assert self.design is not None
        procedures = [region.id for region in self.design.regions if region.kind == "procedure"]
        if requested is not None:
            direct = requested if requested in self.region_by_id else None
            prefixed = f"proc_{requested}" if f"proc_{requested}" in self.region_by_id else None
            if direct is not None and self.region_by_id[direct].kind == "procedure":
                return direct
            if prefixed is not None and self.region_by_id[prefixed].kind == "procedure":
                return prefixed
            raise InterpreterError(f"unknown µhIR procedure '{requested}'")

        roots = [region_id for region_id in procedures if self._is_entry_procedure(region_id)]
        if len(roots) == 1:
            return roots[0]
        if "proc_main" in self.region_by_id and self.region_by_id["proc_main"].kind == "procedure":
            return "proc_main"
        if len(procedures) == 1:
            return procedures[0]
        raise InterpreterError("µhIR design contains multiple procedures; pass --function")

    def _is_entry_procedure(self, region_id: str) -> bool:
        callers = set()
        for region in self.region_by_id.values():
            for node in region.nodes:
                if node.opcode != "call":
                    continue
                child = self._text_attr(node, "child")
                if child is not None:
                    callers.add(child)
        region = self.region_by_id[region_id]
        return region.parent is None and region.kind == "procedure" and region_id not in callers

    def _push_frame(
        self,
        state: ExecutionState,
        region_id: str,
        scalar_arguments: dict[str, int],
        array_arguments: dict[str, dict[str, object]],
    ) -> None:
        saved = getattr(state, "_uhir_frame_stack", [])
        saved.append((dict(state.env), dict(state.value_types)))
        setattr(state, "_uhir_frame_stack", saved)

        state.env = dict(self.global_scalars)
        state.value_types = dict(self.global_scalar_types)
        for name, value in scalar_arguments.items():
            type_hint = self._param_type(region_id, name)
            state.bind_scalar(name, normalize_int(value, type_hint), type_hint)
        if array_arguments:
            state.memory.initialize(array_arguments)

    def _pop_frame(self, state: ExecutionState) -> None:
        stack = getattr(state, "_uhir_frame_stack", None)
        if not stack:
            return
        env, value_types = stack.pop()
        state.env = env
        state.value_types = value_types

    def _param_type(self, region_id: str, name: str) -> Any | None:
        for param in self.procedure_params.get(region_id, ()):
            if param.name == name:
                return param.type_hint
        return None

    def _execute_region(
        self,
        region: UHIRRegion,
        state: ExecutionState,
        *,
        step_limit: int,
        predecessor_label: str | None,
    ) -> _RegionOutcome:
        state.current_block = region.id
        state.predecessor_block = predecessor_label
        nodes = self.ordered_nodes_by_region[region.id]
        index = 0
        current_predecessor = predecessor_label
        while index < len(nodes):
            node = nodes[index]
            if node.opcode == "nop":
                index += 1
                continue

            state.current_block = region.id
            state.predecessor_block = current_predecessor
            self._tick(state, step_limit)

            if self._is_predicated_off(node, state):
                state.record("exec", block=region.id, opcode=node.opcode, detail=f"skip {node.id}")
                index += 1
                continue

            if node.opcode == "phi":
                value = self._resolve_phi(node, current_predecessor, state)
                self._assign_node(region.id, node, value, state)
                state.record("phi", block=region.id, opcode="phi", detail=f"{node.id} = {value}")
                index += 1
                continue

            if node.opcode == "branch":
                outcome, next_index, next_predecessor = self._execute_branch(
                    region,
                    node,
                    state,
                    current_predecessor=current_predecessor,
                    current_index=index,
                    step_limit=step_limit,
                )
                if outcome.returned:
                    return outcome
                index = next_index
                current_predecessor = next_predecessor
                continue

            if node.opcode == "loop":
                outcome = self._execute_loop_node(region, node, state, step_limit=step_limit)
                if outcome.returned:
                    return outcome
                index += 1
                continue

            if node.opcode == "ret":
                value = None if not node.operands else self._resolve_operand(node.operands[0], state)
                value = normalize_int(value, self.procedure_return_types.get(region.id)) if value is not None else None
                state.record("return", block=region.id, opcode="ret", detail=f"return {value}")
                return _RegionOutcome(returned=True, return_value=value)

            self._execute_node(region, node, state, step_limit=step_limit)
            index += 1

        return _RegionOutcome()

    def _execute_branch(
        self,
        region: UHIRRegion,
        node: UHIRNode,
        state: ExecutionState,
        *,
        current_predecessor: str | None,
        current_index: int,
        step_limit: int,
    ) -> tuple[_RegionOutcome, int, str | None]:
        info = self.loop_branches.get((region.id, node.id))
        if info is not None:
            branch_key = (region.id, node.id)
            condition = self._branch_condition(node, state, static_counter_key=branch_key)
            chosen_child = info.body_child_id if condition else info.exit_child_id
            child_outcome = self._execute_child_region(chosen_child, state, step_limit=step_limit)
            if child_outcome.returned:
                return child_outcome, current_index, current_predecessor
            if condition:
                state.record("branch", block=region.id, opcode="branch", detail=f"{node.id} -> {chosen_child} (loop)")
                return _RegionOutcome(), info.header_start_index, info.backedge_label
            state.record("branch", block=region.id, opcode="branch", detail=f"{node.id} -> {chosen_child} (exit)")
            return _RegionOutcome(), current_index + 1, info.exit_label

        condition = self._branch_condition(node, state)
        true_child = self._text_attr(node, "true_child")
        false_child = self._text_attr(node, "false_child")
        if true_child is None or false_child is None:
            raise InterpreterError(f"branch node '{node.id}' is missing true_child/false_child")
        chosen_child = true_child if condition else false_child
        child_outcome = self._execute_child_region(chosen_child, state, step_limit=step_limit)
        if child_outcome.returned:
            return child_outcome, current_index, current_predecessor
        next_predecessor = self._text_attr(node, "true_input_label") if condition else self._text_attr(node, "false_input_label")
        state.record("branch", block=region.id, opcode="branch", detail=f"{node.id} -> {chosen_child}")
        return _RegionOutcome(), current_index + 1, next_predecessor

    def _execute_loop_node(
        self,
        region: UHIRRegion,
        node: UHIRNode,
        state: ExecutionState,
        *,
        step_limit: int,
    ) -> _RegionOutcome:
        child_id = self._text_attr(node, "child")
        if child_id is None:
            raise InterpreterError(f"loop node '{node.id}' is missing child=...")

        explicit = next(
            (
                info
                for info in self.loop_branches.values()
                if info.region_id == child_id
            ),
            None,
        )
        loop_counter_key = None
        if explicit is not None:
            static_trip_count = node.attributes.get("static_trip_count")
            if isinstance(static_trip_count, int):
                loop_counter_key = (explicit.region_id, explicit.branch_id)
                self.active_static_loop_counters[loop_counter_key] = static_trip_count

        try:
            outcome = self._execute_child_region(child_id, state, step_limit=step_limit)
        finally:
            if loop_counter_key is not None:
                self.active_static_loop_counters.pop(loop_counter_key, None)

        state.record("call", block=region.id, opcode="loop", detail=f"{node.id} -> {child_id}")
        return outcome

    def _execute_child_region(
        self,
        child_region_id: str,
        state: ExecutionState,
        *,
        step_limit: int,
    ) -> _RegionOutcome:
        child = self.region_by_id.get(child_region_id)
        if child is None:
            raise InterpreterError(f"unknown child region '{child_region_id}'")
        return self._execute_region(
            child,
            state,
            step_limit=step_limit,
            predecessor_label=self.region_entry_label.get(child_region_id),
        )

    def _execute_node(
        self,
        region: UHIRRegion,
        node: UHIRNode,
        state: ExecutionState,
        *,
        step_limit: int,
    ) -> None:
        opcode = node.opcode
        if opcode == "const":
            if not node.operands:
                raise InterpreterError(f"const node '{node.id}' is missing its literal operand")
            value = normalize_int(self._parse_literal_int(node.operands[0]), normalize_type(node.result_type))
            self._assign_node(region.id, node, value, state)
            state.record("exec", block=region.id, opcode=opcode, detail=f"{node.id} = {value}")
            return

        if opcode in _UNARY_OPS:
            if len(node.operands) != 1:
                raise InterpreterError(f"unary node '{node.id}' expects one operand")
            operand = self._resolve_operand(node.operands[0], state)
            result = eval_unary(opcode, operand, normalize_type(node.result_type))
            self._assign_node(region.id, node, result, state)
            state.record("exec", block=region.id, opcode=opcode, detail=f"{node.id} = {result}")
            return

        if opcode in _BINARY_OPS:
            if len(node.operands) != 2:
                raise InterpreterError(f"binary node '{node.id}' expects two operands")
            lhs = self._resolve_operand(node.operands[0], state)
            rhs = self._resolve_operand(node.operands[1], state)
            result = eval_binary(opcode, lhs, rhs, normalize_type(node.result_type))
            self._assign_node(region.id, node, result, state)
            state.record("exec", block=region.id, opcode=opcode, detail=f"{node.id} = {result}")
            return

        if opcode in _COMPARE_OPS:
            if len(node.operands) != 2:
                raise InterpreterError(f"compare node '{node.id}' expects two operands")
            lhs = self._resolve_operand(node.operands[0], state)
            rhs = self._resolve_operand(node.operands[1], state)
            result = eval_compare(opcode, lhs, rhs)
            self._assign_node(region.id, node, result, state)
            state.record("exec", block=region.id, opcode=opcode, detail=f"{node.id} = {result}")
            return

        if opcode == "sel":
            if len(node.operands) != 3:
                raise InterpreterError(f"sel node '{node.id}' expects condition,true,false operands")
            condition = self._resolve_operand(node.operands[0], state)
            chosen = node.operands[1] if truthy(condition) else node.operands[2]
            result = normalize_int(self._resolve_operand(chosen, state), normalize_type(node.result_type))
            self._assign_node(region.id, node, result, state)
            state.record("exec", block=region.id, opcode=opcode, detail=f"{node.id} = {result}")
            return

        if opcode == "load":
            array_name, index_operand = self._split_mem_access(node, expect_store=False)
            index = self._resolve_operand(index_operand, state)
            raw = state.memory.load(array_name, index)
            type_hint = normalize_type(node.result_type) or state.memory.element_type(array_name)
            result = normalize_int(raw, type_hint)
            self._assign_node(region.id, node, result, state)
            state.record("exec", block=region.id, opcode=opcode, detail=f"{node.id} = {array_name}[{int(index)}] -> {result}")
            return

        if opcode == "store":
            array_name, index_operand, value_operand = self._split_mem_access(node, expect_store=True)
            index = self._resolve_operand(index_operand, state)
            value = self._resolve_operand(value_operand, state)
            stored = normalize_int(value, state.memory.element_type(array_name))
            state.memory.store(array_name, index, stored)
            state.record("exec", block=region.id, opcode=opcode, detail=f"{array_name}[{int(index)}] = {stored}")
            return

        if opcode == "call":
            result = self._execute_call(region, node, state, step_limit=step_limit)
            if node.result_type is not None:
                normalized = normalize_int(0 if result is None else result, normalize_type(node.result_type))
                self._assign_node(region.id, node, normalized, state)
            elif result is not None:
                self._assign_node(region.id, node, result, state)
            detail = f"{node.id} = {self._call_name(node)}(...) -> {result}" if result is not None else f"{self._call_name(node)}(...)"
            state.record("exec", block=region.id, opcode=opcode, detail=detail)
            return

        if opcode == "print":
            rendered = self._format_print(node, state)
            state.stdout.append(rendered)
            state.record("exec", block=region.id, opcode=opcode, detail=rendered)
            return

        raise InterpreterError(f"unsupported µhIR opcode '{opcode}'")

    def _execute_call(
        self,
        region: UHIRRegion,
        node: UHIRNode,
        state: ExecutionState,
        *,
        step_limit: int,
    ) -> int | None:
        child_id = self._text_attr(node, "child")
        if child_id is None:
            raise InterpreterError(f"call node '{node.id}' is missing child=...")
        callee = self.region_by_id.get(child_id)
        if callee is None or callee.kind != "procedure":
            raise InterpreterError(f"call node '{node.id}' references unknown procedure '{child_id}'")

        actual_operands = list(node.operands[1:] if node.operands else [])
        params = list(self.procedure_params.get(child_id, ()))
        if len(actual_operands) != len(params):
            raise InterpreterError(
                f"call to '{self._call_name(node)}' expected {len(params)} operands, got {len(actual_operands)}"
            )

        scalar_arguments: dict[str, int] = {}
        array_arguments: dict[str, dict[str, object]] = {}
        for param, operand in zip(params, actual_operands, strict=True):
            if param.kind == "array":
                array_name = self._resolve_array_name(operand)
                if not state.memory.has(array_name):
                    raise InterpreterError(
                        f"call to '{self._call_name(node)}' received unknown array '{array_name}' for '{param.name}'"
                    )
                array_arguments[param.name] = {"alias": array_name}
                continue
            scalar_arguments[param.name] = self._resolve_operand(operand, state)

        self._push_frame(state, child_id, scalar_arguments, array_arguments)
        try:
            outcome = self._execute_region(
                callee,
                state,
                step_limit=step_limit,
                predecessor_label=self.region_entry_label.get(child_id),
            )
        finally:
            self._pop_frame(state)
        state.record("call", block=region.id, opcode="call", detail=f"{region.id} -> {child_id} returned {outcome.return_value}")
        return outcome.return_value

    def _resolve_phi(self, node: UHIRNode, predecessor_label: str | None, state: ExecutionState) -> int:
        incoming = node.attributes.get("incoming")
        if not isinstance(incoming, tuple) or len(incoming) != len(node.operands):
            raise InterpreterError(f"phi node '{node.id}' is missing valid incoming=[...] metadata")
        if predecessor_label is None:
            raise InterpreterError(f"phi node '{node.id}' requires one predecessor label")
        for label, operand in zip(incoming, node.operands, strict=True):
            if label == predecessor_label:
                return normalize_int(self._resolve_operand(operand, state), normalize_type(node.result_type))
        raise InterpreterError(
            f"phi node '{node.id}' has no incoming value for predecessor '{predecessor_label}'"
        )

    def _assign_node(self, region_id: str, node: UHIRNode, value: int, state: ExecutionState) -> None:
        type_hint = normalize_type(node.result_type)
        for name in self.node_mappings.get((region_id, node.id), (node.id,)):
            state.bind_scalar(name, value, type_hint)

    def _resolve_operand(self, operand: str, state: ExecutionState) -> int:
        literal = self._try_parse_literal(operand)
        if literal is not None:
            value, type_hint = literal
            return normalize_int(value, type_hint)
        return state.read_scalar(operand)

    def _resolve_array_name(self, operand: str) -> str:
        if self._try_parse_literal(operand) is not None:
            raise InterpreterError(f"array operands must be symbolic names, got literal '{operand}'")
        return operand

    def _branch_condition(
        self,
        node: UHIRNode,
        state: ExecutionState,
        *,
        static_counter_key: tuple[str, str] | None = None,
    ) -> bool:
        if node.operands:
            return truthy(self._resolve_operand(node.operands[0], state))
        if static_counter_key is not None and static_counter_key in self.active_static_loop_counters:
            remaining = self.active_static_loop_counters[static_counter_key]
            if remaining <= 0:
                return False
            self.active_static_loop_counters[static_counter_key] = remaining - 1
            return True
        raise InterpreterError(f"branch node '{node.id}' has no condition operand")

    def _is_predicated_off(self, node: UHIRNode, state: ExecutionState) -> bool:
        predicate = node.attributes.get("pred")
        if not isinstance(predicate, str) or not predicate:
            return False
        if predicate.startswith("!"):
            return truthy(self._resolve_operand(predicate[1:], state))
        return not truthy(self._resolve_operand(predicate, state))

    def _format_print(self, node: UHIRNode, state: ExecutionState) -> str:
        format_text = self._text_attr(node, "format")
        values = [self._resolve_operand(operand, state) for operand in node.operands]
        if format_text is None:
            return " ".join(str(value) for value in values)
        if not values:
            return format_text
        try:
            return format_text % tuple(values)
        except (TypeError, ValueError) as exc:
            raise InterpreterError(f"invalid print format {format_text!r}") from exc

    def _split_mem_access(self, node: UHIRNode, *, expect_store: bool) -> tuple[str, ...]:
        if not node.operands:
            raise InterpreterError(f"{node.opcode} node '{node.id}' is missing operands")
        head = node.operands[0]
        match = re.fullmatch(r"([A-Za-z_][\w$]*)\[(.+)\]", head)
        if match is not None:
            array_name = match.group(1)
            index_text = match.group(2).strip()
            if expect_store:
                if len(node.operands) != 2:
                    raise InterpreterError(f"store node '{node.id}' expects two operands")
                return (array_name, index_text, node.operands[1])
            if len(node.operands) != 1:
                raise InterpreterError(f"load node '{node.id}' expects one array[index] operand")
            return (array_name, index_text)

        if expect_store and len(node.operands) == 3:
            return (node.operands[0], node.operands[1], node.operands[2])
        if not expect_store and len(node.operands) == 2:
            return (node.operands[0], node.operands[1])
        raise InterpreterError(f"malformed {node.opcode} node '{node.id}'")

    def _infer_procedure_params(self, region: UHIRRegion) -> tuple[_ProcedureParam, ...]:
        produced = {node.id for node in region.nodes}
        produced.update(mapping.source_id for mapping in region.mappings)
        ordered: list[_ProcedureParam] = []
        seen: set[str] = set()

        def add_param(name: str, kind: str, type_hint: Any | None = None) -> None:
            if (
                not name
                or name in seen
                or name in produced
                or name in self.global_scalars
                or self._try_parse_literal(name) is not None
            ):
                return
            seen.add(name)
            ordered.append(_ProcedureParam(name, kind, type_hint))

        for node in self.ordered_nodes_by_region[region.id]:
            predicate = node.attributes.get("pred")
            if isinstance(predicate, str) and predicate:
                add_param(predicate[1:] if predicate.startswith("!") else predicate, "scalar")

            if node.opcode == "call":
                for operand in node.operands[1:]:
                    add_param(operand, "scalar")
                continue
            if node.opcode == "load":
                array_name, index_operand = self._split_mem_access(node, expect_store=False)
                add_param(array_name, "array")
                add_param(index_operand, "scalar")
                continue
            if node.opcode == "store":
                array_name, index_operand, value_operand = self._split_mem_access(node, expect_store=True)
                add_param(array_name, "array")
                add_param(index_operand, "scalar")
                add_param(value_operand, "scalar")
                continue
            if node.opcode in {"branch", "ret", "print", "phi", "sel", "const", *tuple(_UNARY_OPS), *tuple(_BINARY_OPS), *tuple(_COMPARE_OPS)}:
                for operand in node.operands:
                    add_param(operand, "scalar")
        return tuple(ordered)

    def _infer_loop_entry_label(
        self,
        region: UHIRRegion,
        header_node_ids: Iterable[str],
        *,
        excluded_labels: set[str],
    ) -> str | None:
        candidates: set[str] = set()
        for node in region.nodes:
            if node.id not in header_node_ids or node.opcode != "phi":
                continue
            incoming = node.attributes.get("incoming")
            if not isinstance(incoming, tuple):
                continue
            filtered = [label for label in incoming if label not in excluded_labels]
            if len(filtered) == 1:
                candidates.add(filtered[0])
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    def _port_param_kind(self, type_text: str) -> tuple[str, Any | None]:
        match = _MEMREF_RE.fullmatch(type_text.strip())
        if match is None:
            return "scalar", normalize_type(type_text)
        element_type, extent_text = match.groups()
        extent = None if extent_text is None else int(extent_text)
        return "array", ArrayType(normalize_type(element_type), extent)

    def _try_parse_literal(self, text: str) -> tuple[int, Any | None] | None:
        typed = _TYPED_LITERAL_RE.fullmatch(text)
        if typed is not None:
            value_text, type_text = typed.groups()
            return int(value_text), normalize_type(type_text)
        if re.fullmatch(r"[+-]?[0-9]+", text):
            return int(text), None
        return None

    def _parse_literal_int(self, text: str) -> int:
        literal = self._try_parse_literal(text)
        if literal is None:
            raise InterpreterError(f"expected integer literal, got '{text}'")
        return literal[0]

    @staticmethod
    def _call_name(node: UHIRNode) -> str:
        if node.operands:
            return node.operands[0]
        return node.id

    @staticmethod
    def _text_attr(node: UHIRNode, name: str) -> str | None:
        value = node.attributes.get(name)
        return value if isinstance(value, str) else None

    @staticmethod
    def _tick(state: ExecutionState, step_limit: int) -> None:
        state.steps += 1
        if state.steps > step_limit:
            raise InterpreterError(f"step limit exceeded ({step_limit})")


def run_uhir(
    design: UHIRDesign,
    arguments: dict[str, int] | None = None,
    arrays: dict[str, dict[str, object]] | None = None,
    *,
    function_name: str | None = None,
    trace: bool = False,
    step_limit: int = 100_000,
    state: ExecutionState | None = None,
) -> ExecutionResult:
    """Execute one seq-style µhIR design."""
    return UHIRInterpreter().run(
        design,
        arguments=arguments,
        arrays=arrays,
        function_name=function_name,
        trace=trace,
        step_limit=step_limit,
        state=state,
    )


__all__ = ["UHIRInterpreter", "run_uhir"]
