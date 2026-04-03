"""Shared pass-manager framework for custom µhLS pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import Parameter as InspectParameter
from inspect import signature
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class PassContext:
    """Mutable shared state threaded through one pass pipeline."""

    analyses: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    pass_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class PassRunResult:
    """Result bundle returned by ``PassManager.run_with_context``."""

    output: Any
    context: PassContext


@runtime_checkable
class Pass(Protocol):
    """Protocol for class-based transformation or analysis passes."""

    name: str

    def run(self, ir: Any, context: PassContext) -> Any:
        """Apply the pass to ``ir`` and return the transformed value."""


@dataclass(frozen=True)
class AnalysisPass:
    """Adapter that stores reusable analysis results in the pass context."""

    name: str
    analysis: Callable[[Any], Any]
    key: str | None = None

    def run(self, ir: Any, context: PassContext) -> Any:
        context.analyses[self.key or self.name] = self.analysis(ir)
        return ir


@dataclass(frozen=True)
class ModuleTransformPass:
    """Generic wrapper for passes that transform modules only."""

    name: str
    transform: Callable[[Any], Any]

    def run(self, ir: Any, context: PassContext) -> Any:
        return self.transform(ir)


@dataclass(frozen=True)
class FunctionOrModuleTransformPass:
    """Generic wrapper for passes that accept either functions or modules."""

    name: str
    function_transform: Callable[[Any], Any]
    module_transform: Callable[[Any], Any]

    def run(self, ir: Any, context: PassContext) -> Any:
        if hasattr(ir, "functions"):
            return self.module_transform(ir)
        return self.function_transform(ir)


@dataclass
class PassManager:
    """Apply a sequence of passes to one IR object."""

    pipeline: list[Any]

    def run(self, ir: Any, context: PassContext | None = None) -> Any:
        """Run the pipeline and return the transformed IR."""
        return self.run_with_context(ir, context).output

    def run_with_context(self, ir: Any, context: PassContext | None = None) -> PassRunResult:
        """Run the pipeline and return both output and pass context."""
        active_context = context or PassContext()
        current = ir
        for pass_like in self.pipeline:
            name = pass_name(pass_like)
            current = _invoke_pass(pass_like, current, active_context)
            active_context.history.append(name)
        return PassRunResult(output=current, context=active_context)


def pass_name(pass_like: Any) -> str:
    """Return a stable display name for one pass-like object."""
    explicit = getattr(pass_like, "name", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    if hasattr(pass_like, "__name__"):
        return str(pass_like.__name__)
    return pass_like.__class__.__name__


def analysis_pass(name: str, analysis: Callable[[Any], Any], key: str | None = None) -> AnalysisPass:
    """Create a pass wrapper for one reusable analysis function."""
    return AnalysisPass(name=name, analysis=analysis, key=key)


def module_pass(name: str, transform: Callable[[Any], Any]) -> ModuleTransformPass:
    """Create a generic class-free module transform pass."""
    return ModuleTransformPass(name=name, transform=transform)


def function_or_module_pass(
    name: str,
    function_transform: Callable[[Any], Any],
    module_transform: Callable[[Any], Any],
) -> FunctionOrModuleTransformPass:
    """Create a generic class-free function-or-module transform pass."""
    return FunctionOrModuleTransformPass(
        name=name,
        function_transform=function_transform,
        module_transform=module_transform,
    )


def _invoke_pass(pass_like: Any, ir: Any, context: PassContext) -> Any:
    if isinstance(pass_like, Pass):
        return _invoke_callable(pass_like.run, ir, context)

    if callable(pass_like):
        return _invoke_callable(pass_like, ir, context)

    raise TypeError(f"unsupported pass object {pass_like!r}")


def _invoke_callable(pass_like: Callable[..., Any], ir: Any, context: PassContext) -> Any:
    params = list(signature(pass_like).parameters.values())
    positional = [
        param
        for param in params
        if param.kind in {InspectParameter.POSITIONAL_ONLY, InspectParameter.POSITIONAL_OR_KEYWORD}
    ]
    required = [param for param in positional if param.default is InspectParameter.empty]

    if len(required) <= 1 and len(positional) <= 1:
        return pass_like(ir)
    if len(required) <= 2 and len(positional) <= 2:
        return pass_like(ir, context)
    if len(required) <= 3 and len(positional) <= 3:
        return pass_like(ir, context, context.pass_args)
    raise TypeError(
        f"callable pass '{pass_name(pass_like)}' must accept (ir), (ir, context), or (ir, context, pass_args)"
    )
