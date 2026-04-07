"""Generic uglir-to-RTL dispatch."""

from __future__ import annotations

from dataclasses import dataclass

from ..uglir import UGLIRDesign, validate_uglir_for_rtl
from .verilog import emit_uglir_to_verilog

RTL_HDLS: tuple[str, ...] = ("verilog", "vhdl", "systemc")
RTL_RESETS: tuple[str, ...] = (
    "sync+active_hi",
    "sync+active_lo",
    "async+active_hi",
    "async+active_lo",
)


@dataclass(frozen=True)
class ResetSpec:
    kind: str
    polarity: str


def lower_uglir_to_rtl(
    design: UGLIRDesign,
    hdl: str,
    reset: str = "sync+active_hi",
) -> str:
    """Lower one uglir design to one textual HDL artifact."""
    if design.stage != "uglir":
        raise ValueError(f"rtl lowering expects uglir input, got stage '{design.stage}'")
    validate_uglir_for_rtl(design)
    normalized = hdl.strip().lower()
    reset_spec = parse_reset_spec(reset)
    if normalized == "verilog":
        return emit_uglir_to_verilog(design, reset=reset_spec)
    if normalized in RTL_HDLS:
        raise NotImplementedError(f"rtl lowering for HDL '{normalized}' is not implemented yet")
    raise ValueError(f"unsupported HDL '{hdl}'")


def parse_reset_spec(text: str) -> ResetSpec:
    normalized = text.strip().lower()
    if not normalized:
        raise ValueError("unsupported reset ''")
    parts = tuple(part.strip() for part in normalized.split("+") if part.strip())
    if len(parts) != 2:
        raise ValueError(
            f"unsupported reset '{text}'; expected <sync|async>+<active_hi|active_lo>"
        )
    kind, polarity = parts
    if kind not in {"sync", "async"}:
        raise ValueError(
            f"unsupported reset '{text}'; expected <sync|async>+<active_hi|active_lo>"
        )
    polarity_aliases = {
        "active_hi": "active_hi",
        "active_high": "active_hi",
        "active_lo": "active_lo",
        "active_low": "active_lo",
    }
    normalized_polarity = polarity_aliases.get(polarity)
    if normalized_polarity is None:
        raise ValueError(
            f"unsupported reset '{text}'; expected <sync|async>+<active_hi|active_lo>"
        )
    return ResetSpec(kind=kind, polarity=normalized_polarity)
