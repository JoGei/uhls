"""Runtime state shared by the µIR interpreter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .memory import ArrayMemory


class InterpreterError(RuntimeError):
    """Raised when the IR or runtime state cannot be executed."""


@dataclass(frozen=True)
class TraceEvent:
    """One trace entry emitted during interpretation."""

    step: int
    kind: str
    block: str | None = None
    opcode: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class CallHookResult:
    """One optional foreign-call result supplied by an execution hook."""

    return_value: int | None
    updated_arrays: dict[str, list[int]] = field(default_factory=dict)
    stdout: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionState:
    """Mutable execution state for a single function activation."""

    env: dict[str, int] = field(default_factory=dict)
    value_types: dict[str, Any] = field(default_factory=dict)
    memory: ArrayMemory = field(default_factory=ArrayMemory)
    pending_params: list[tuple[int, object]] = field(default_factory=list)
    trace_enabled: bool = False
    trace: list[TraceEvent] = field(default_factory=list)
    stdout: list[str] = field(default_factory=list)
    steps: int = 0
    current_block: str | None = None
    predecessor_block: str | None = None
    returned: bool = False
    return_value: int | None = None

    def bind_scalar(self, name: str, value: int, type_hint: Any | None = None) -> None:
        """Bind or overwrite one scalar name in the environment.

        Args:
            name: IR scalar symbol to bind.
            value: Integer value assigned to the symbol.
            type_hint: Optional type metadata retained for later inspection.
        """
        self.env[name] = int(value)
        if type_hint is not None:
            self.value_types[name] = type_hint

    def read_scalar(self, name: str) -> int:
        """Read one scalar from the environment.

        Args:
            name: IR scalar symbol to resolve.

        Returns:
            The current integer value bound to ``name``.
        """
        try:
            return self.env[name]
        except KeyError as exc:
            raise InterpreterError(f"use of undefined scalar '{name}'") from exc

    def scalar_type(self, name: str) -> Any | None:
        """Return the remembered type metadata for ``name`` if available."""
        return self.value_types.get(name)

    def record(
        self,
        kind: str,
        *,
        block: str | None = None,
        opcode: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Append one trace event when tracing is enabled.

        Args:
            kind: High-level trace category such as ``exec`` or ``branch``.
            block: Block label associated with the event.
            opcode: Instruction opcode associated with the event.
            detail: Human-readable detail string for debugging.
        """
        if not self.trace_enabled:
            return
        self.trace.append(
            TraceEvent(
                step=self.steps,
                kind=kind,
                block=block,
                opcode=opcode,
                detail=detail,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the current execution state."""
        return {
            "env": dict(self.env),
            "value_types": dict(self.value_types),
            "memory": self.memory.snapshot(),
            "pending_params": list(self.pending_params),
            "stdout": list(self.stdout),
            "steps": self.steps,
            "current_block": self.current_block,
            "predecessor_block": self.predecessor_block,
            "returned": self.returned,
            "return_value": self.return_value,
        }


@dataclass(frozen=True)
class ExecutionResult:
    """Interpreter return bundle."""

    return_value: int | None
    state: ExecutionState
