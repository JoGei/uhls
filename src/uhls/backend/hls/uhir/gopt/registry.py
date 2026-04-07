"""Registry for built-in µhIR graph optimizer passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .builtin import FoldPredicatesPass, InferLoopsPass, InferStaticPass, LoopDialectPass, PredicatePass, SimplifyStaticControlPass


@dataclass(frozen=True)
class GOptPassSpec:
    """Metadata for one built-in µhIR graph optimization pass."""

    name: str
    factory: Callable[[], object]
    description: str
    example: str
    aliases: tuple[str, ...] = ()


_GOPT_PASS_SPECS: tuple[GOptPassSpec, ...] = (
    GOptPassSpec(
        name="infer_loops",
        factory=InferLoopsPass,
        description="Annotate explicit loop hierarchy with loop ids, header markers, and backedge facts.",
        example="uhls gopt input.seq.uhir -p infer_loops -o output.seq.uhir",
        aliases=("loops",),
    ),
    GOptPassSpec(
        name="translate_loop_dialect",
        factory=LoopDialectPass,
        description="Own explicit loop-dialect shaping on seq-stage µhIR; today it preserves explicit loop hierarchy.",
        example="uhls gopt input.seq.uhir -p infer_loops,translate_loop_dialect -o output.seq.uhir",
        aliases=("loop_dialect",),
    ),
    GOptPassSpec(
        name="infer_static",
        factory=InferStaticPass,
        description="Infer static loop-trip facts on seq-stage µhIR and annotate loop hierarchy nodes.",
        example="uhls gopt input.seq.uhir -p infer_static -o output.seq.uhir",
        aliases=("infer",),
    ),
    GOptPassSpec(
        name="simplify_static_control",
        factory=SimplifyStaticControlPass,
        description="Consume inferred static control facts and simplify seq-stage hierarchy conservatively.",
        example="uhls gopt input.seq.uhir -p infer_static,simplify_static_control -o output.seq.uhir",
        aliases=("simplify_static",),
    ),
    GOptPassSpec(
        name="predicate",
        factory=PredicatePass,
        description="Conservatively predicate pure branch-export dataflow into the parent seq region.",
        example="uhls gopt input.seq.uhir -p predicate -o output.seq.uhir",
    ),
    GOptPassSpec(
        name="fold_predicates",
        factory=FoldPredicatesPass,
        description="Fold redundant complementary predicated computations and trivial select nodes.",
        example="uhls gopt input.seq.uhir -p predicate,fold_predicates -o output.seq.uhir",
    ),
)


def builtin_gopt_specs() -> tuple[GOptPassSpec, ...]:
    """Return built-in µhIR graph optimizer pass metadata."""
    return _GOPT_PASS_SPECS


def builtin_gopt_pass_names() -> list[str]:
    """Return built-in µhIR graph optimizer pass names."""
    return [spec.name for spec in _GOPT_PASS_SPECS]


def create_builtin_gopt_pass(name: str) -> object:
    """Instantiate one built-in µhIR graph optimizer pass."""
    normalized = name.strip().lower().replace("-", "_")
    for spec in _GOPT_PASS_SPECS:
        if normalized == spec.name or normalized in spec.aliases:
            return spec.factory()
    supported = ", ".join(spec.name for spec in _GOPT_PASS_SPECS)
    raise ValueError(f"unknown µhIR graph optimization pass '{name}'; expected one of: {supported}")
