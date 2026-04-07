"""Wrapper/protocol composition for µglIR designs."""

from __future__ import annotations

from uhls.backend.hls.uhir.model import UHIRDesign

from ..rtl.protocol import build_wishbone_slave_wrapper_uglir, plan_obi_slave_protocol, plan_wishbone_slave_protocol
from ..rtl.wrap import plan_master_wrapper, plan_slave_wrapper
from .validate import validate_uglir_for_rtl

GLUE_WRAPS: tuple[str, ...] = ("none", "slave", "master")
GLUE_PROTOCOLS: tuple[str, ...] = ("memory", "wishbone", "obi")


def wrap_uglir_design(design: UHIRDesign, wrap: str | None = None, protocol: str | None = None) -> UHIRDesign:
    """Apply one optional wrapper/protocol composition to one µglIR design."""
    if design.stage != "uglir":
        raise ValueError(f"µglIR wrapping expects µglIR input, got stage '{design.stage}'")
    validate_uglir_for_rtl(design)
    normalized_wrap = _normalize_optional_choice("wrap", wrap, GLUE_WRAPS)
    normalized_protocol = _normalize_optional_choice("protocol", protocol, GLUE_PROTOCOLS)
    if (normalized_wrap is None) != (normalized_protocol is None):
        raise ValueError("µglIR wrapping requires --wrap and --protocol together")
    if normalized_wrap is None:
        return design
    if normalized_wrap == "none" and normalized_protocol == "memory":
        return design
    if normalized_wrap == "slave" and normalized_protocol == "wishbone":
        wrapper_plan = plan_slave_wrapper(design, normalized_protocol)
        protocol_plan = plan_wishbone_slave_protocol(wrapper_plan)
        wrapped = build_wishbone_slave_wrapper_uglir(design, wrapper_plan, protocol_plan)
        validate_uglir_for_rtl(wrapped)
        return wrapped
    if normalized_wrap == "slave" and normalized_protocol == "obi":
        wrapper_plan = plan_slave_wrapper(design, normalized_protocol)
        plan_obi_slave_protocol(wrapper_plan)
        raise NotImplementedError("µglIR wrapping for wrap='slave' protocol='obi' is not implemented yet")
    if normalized_wrap == "master":
        plan_master_wrapper(design, normalized_protocol or "")
        raise NotImplementedError(
            f"µglIR wrapping for wrap='{normalized_wrap}' protocol='{normalized_protocol}' is not implemented yet"
        )
    raise NotImplementedError(
        f"µglIR wrapping for wrap='{normalized_wrap}' protocol='{normalized_protocol}' is not implemented yet"
    )


def _normalize_optional_choice(name: str, value: str | None, choices: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"unsupported {name} '{value}'")
    return normalized
