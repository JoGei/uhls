"""Built-in µhIR graph optimizer passes."""

from .infer_static import InferStaticPass
from .predicate import PredicatePass
from .simplify_static_control import SimplifyStaticControlPass

__all__ = [
    "InferStaticPass",
    "PredicatePass",
    "SimplifyStaticControlPass",
]
