"""µglIR stage lowering and validation facade."""

from __future__ import annotations

from .uglir import GLUE_PROTOCOLS, GLUE_WRAPS, lower_fsm_to_uglir, validate_uglir_for_rtl, wrap_uglir_design

__all__ = ["GLUE_PROTOCOLS", "GLUE_WRAPS", "lower_fsm_to_uglir", "validate_uglir_for_rtl", "wrap_uglir_design"]
