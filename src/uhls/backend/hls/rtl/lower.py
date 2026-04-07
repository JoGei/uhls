"""Generic uglir-to-RTL dispatch."""

from __future__ import annotations

from uhls.backend.hls.uhir.model import UHIRDesign

from ..uglir import validate_uglir_for_rtl
from .verilog import emit_uglir_to_verilog, emit_uglir_to_verilog_wrapped

RTL_HDLS: tuple[str, ...] = ("verilog", "vhdl", "systemc")
RTL_WRAPS: tuple[str, ...] = ("none", "slave", "master")
RTL_PROTOCOLS: tuple[str, ...] = ("memory", "wishbone", "obi")


def lower_uglir_to_rtl(
    design: UHIRDesign,
    hdl: str,
    wrap: str | None = None,
    protocol: str | None = None,
) -> str:
    """Lower one uglir design to one textual HDL artifact."""
    if design.stage != "uglir":
        raise ValueError(f"rtl lowering expects uglir input, got stage '{design.stage}'")
    validate_uglir_for_rtl(design)
    normalized = hdl.strip().lower()
    normalized_wrap = _normalize_optional_choice("wrap", wrap, RTL_WRAPS)
    normalized_protocol = _normalize_optional_choice("protocol", protocol, RTL_PROTOCOLS)
    if (normalized_wrap is None) != (normalized_protocol is None):
        raise ValueError("rtl lowering requires --wrap and --protocol together")
    if normalized == "verilog":
        if normalized_wrap == "none" and normalized_protocol == "memory":
            return emit_uglir_to_verilog(design)
        if normalized_wrap is None:
            return emit_uglir_to_verilog(design)
        if normalized_wrap == "slave" and normalized_protocol == "wishbone":
            return emit_uglir_to_verilog_wrapped(design, wrap=normalized_wrap, protocol=normalized_protocol)
        if normalized_wrap in RTL_WRAPS and normalized_protocol in RTL_PROTOCOLS:
            raise NotImplementedError(
                f"rtl lowering stub for wrap='{normalized_wrap}' protocol='{normalized_protocol}' is not implemented yet"
            )
        return emit_uglir_to_verilog(design)
    if normalized in RTL_HDLS:
        raise NotImplementedError(f"rtl lowering for HDL '{normalized}' is not implemented yet")
    raise ValueError(f"unsupported HDL '{hdl}'")


def _normalize_optional_choice(name: str, value: str | None, choices: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"unsupported {name} '{value}'")
    return normalized
