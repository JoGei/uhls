"""Built-in µhIR graph optimizer passes."""

from .fold_predicates import FoldPredicatesPass
from .infer_static import InferStaticPass
from .infer_loops import InferLoopsPass
from .loop_dialect import LoopDialectPass
from .predicate import PredicatePass
from .simplify_static_control import SimplifyStaticControlPass

__all__ = [
    "FoldPredicatesPass",
    "InferStaticPass",
    "InferLoopsPass",
    "LoopDialectPass",
    "PredicatePass",
    "SimplifyStaticControlPass",
]
