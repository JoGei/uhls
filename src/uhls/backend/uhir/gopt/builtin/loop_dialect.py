"""Loop-dialect pass scaffold for seq-stage µhIR."""

from __future__ import annotations

from copy import deepcopy

from uhls.backend.uhir.model import UHIRDesign
from uhls.backend.uhir.gopt.loops import collect_explicit_loops, collect_loop_candidates, explicit_loop_from_candidate


class LoopDialectPass:
    """Own explicit loop-dialect shape for seq-stage µhIR."""

    name = "translate_loop_dialect"

    def run(self, ir: UHIRDesign) -> UHIRDesign:
        if ir.stage != "seq":
            raise ValueError(f"translate_loop_dialect expects seq-stage µhIR, got stage '{ir.stage}'")

        result = deepcopy(ir)
        if collect_explicit_loops(result):
            return result
        for candidate in collect_loop_candidates(result):
            result = explicit_loop_from_candidate(result, candidate)
        return result
