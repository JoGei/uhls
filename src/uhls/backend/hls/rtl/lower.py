"""Generic uglir-to-RTL dispatch."""

from __future__ import annotations

from uhls.backend.hls.uhir.model import UHIRDesign

from ..uglir import validate_uglir_for_rtl
from .verilog import emit_uglir_to_verilog

RTL_HDLS: tuple[str, ...] = ("verilog", "vhdl", "systemc")


def lower_uglir_to_rtl(
    design: UHIRDesign,
    hdl: str,
) -> str:
    """Lower one uglir design to one textual HDL artifact."""
    if design.stage != "uglir":
        raise ValueError(f"rtl lowering expects uglir input, got stage '{design.stage}'")
    validate_uglir_for_rtl(design)
    normalized = hdl.strip().lower()
    if normalized == "verilog":
        return emit_uglir_to_verilog(design)
    if normalized in RTL_HDLS:
        raise NotImplementedError(f"rtl lowering for HDL '{normalized}' is not implemented yet")
    raise ValueError(f"unsupported HDL '{hdl}'")
