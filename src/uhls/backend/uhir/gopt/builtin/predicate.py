"""Predication-oriented control simplification placeholder."""

from __future__ import annotations

from uhls.backend.uhir.model import UHIRDesign


class PredicatePass:
    """Future predication pass for branch-hierarchy simplification."""

    name = "predicate"

    def run(self, ir: UHIRDesign) -> UHIRDesign:
        # TODO: Implement µhIR-level predication once the static control
        # pipeline is fully stabilized and the desired transformed canonical
        # form is specified.
        raise NotImplementedError("predicate is registered but not implemented yet")
