"""µIR interpreter adapter with block-entry phi resolution."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .base import BaseAdapter, InterpreterBase
from .runtime import CallHookResult, ExecutionResult, ExecutionState, InterpreterError


class UIRAdapter(BaseAdapter):
    """Canonical µIR adapter that resolves block-leading phis on entry."""

    def on_block_entry(
        self,
        interpreter: InterpreterBase,
        function: Any,
        block: Any,
        state: ExecutionState,
        predecessor: str | None,
    ) -> None:
        """Resolve all leading phi nodes for the predecessor edge just taken."""
        body, _ = self.split_block(block)
        phis: list[Any] = []
        rest_started = False

        for instruction in body:
            if self.opcode(instruction) == "phi" and not rest_started:
                phis.append(instruction)
                continue
            rest_started = True
            if self.opcode(instruction) == "phi":
                raise InterpreterError(
                    f"phi must appear before non-phi instructions in block '{self.block_label(block)}'"
                )

        if not phis:
            return

        if predecessor is None:
            raise InterpreterError(
                f"cannot resolve phi nodes in entry block '{self.block_label(block)}' without a predecessor"
            )

        updates: list[tuple[str, int, Any]] = []
        for phi in phis:
            matched = False
            for pred_label, operand in self.phi_incomings(phi):
                if pred_label != predecessor:
                    continue
                value = interpreter._resolve_operand(operand, state)
                updates.append((self.result_name(phi), value, self.result_type(phi)))
                matched = True
                break

            if not matched:
                raise InterpreterError(
                    f"phi in block '{self.block_label(block)}' has no incoming value for predecessor '{predecessor}'"
                )

        for name, value, type_hint in updates:
            normalized = value if type_hint is None else interpreter._assign_phi_value(value, type_hint)
            state.bind_scalar(name, normalized, type_hint)
            state.record(
                "phi",
                block=self.block_label(block),
                opcode="phi",
                detail=f"{name} = {normalized} from {predecessor}",
            )

    def body_instructions(self, block: Any) -> list[Any]:
        """Return the executable body of a block with leading phis removed."""
        body, _ = self.split_block(block)
        result: list[Any] = []
        dropping_phis = True
        for instruction in body:
            if dropping_phis and self.opcode(instruction) == "phi":
                continue
            dropping_phis = False
            result.append(instruction)
        return result


class UIRInterpreter(InterpreterBase):
    """Interpreter for canonical µIR."""

    def __init__(
        self,
        *,
        call_hook: Callable[[str, Any, dict[str, int], dict[str, dict[str, object]], ExecutionState], CallHookResult | None]
        | None = None,
    ) -> None:
        """Create a µIR interpreter using the SSA-aware adapter."""
        super().__init__(adapter=UIRAdapter(), call_hook=call_hook)

    def _assign_phi_value(self, value: int, type_hint: Any | None) -> int:
        """Normalize a selected phi incoming value to the phi result type."""
        from .eval import normalize_int

        return normalize_int(value, type_hint)


def run_uir(
    function: Any,
    arguments: Mapping[str, Any] | Sequence[Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
    module: Any | None = None,
    *,
    trace: bool = False,
    step_limit: int = 100_000,
    state: ExecutionState | None = None,
    call_hook: Callable[[str, Any, dict[str, int], dict[str, dict[str, object]], ExecutionState], CallHookResult | None]
    | None = None,
) -> ExecutionResult:
    """Execute one µIR function with explicit scalar and array inputs."""
    return UIRInterpreter(call_hook=call_hook).run(
        function,
        arguments,
        arrays,
        module,
        trace=trace,
        step_limit=step_limit,
        state=state,
    )
