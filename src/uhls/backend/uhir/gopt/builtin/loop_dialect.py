"""Loop-dialect pass scaffold for seq-stage µhIR."""

from __future__ import annotations

from copy import deepcopy

from uhls.backend.uhir.model import UHIRDesign
from uhls.backend.uhir.gopt.loops import collect_explicit_loops


class LoopDialectPass:
    """Own explicit loop-dialect shape for seq-stage µhIR."""

    name = "loop_dialect"

    def run(self, ir: UHIRDesign) -> UHIRDesign:
        if ir.stage != "seq":
            raise ValueError(f"loop_dialect expects seq-stage µhIR, got stage '{ir.stage}'")

        result = deepcopy(ir)

        # Today seq still materializes explicit loop hierarchy directly.
        # Keep this pass as a stable ownership boundary now, then teach it to
        # rewrite branch/backedge-only seq-stage µhIR into explicit loop/body/
        # empty regions once seq becomes a pure structural importer.
        _ = collect_explicit_loops(result)
        return result
