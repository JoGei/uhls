"""Wrapper/protocol composition for µglIR designs."""

from __future__ import annotations

from ..rtl.protocol import (
    build_obi_slave_wrapper_uglir,
    build_wishbone_slave_wrapper_uglir,
    parse_protocol_spec,
    plan_obi_slave_protocol,
    plan_wishbone_slave_protocol,
)
from ..rtl.wrap import plan_master_wrapper, plan_slave_wrapper
from .model import UGLIRDesign, to_uglir_design
from .validate import validate_uglir_for_rtl

GLUE_WRAPS: tuple[str, ...] = ("none", "slave", "master")
GLUE_PROTOCOLS: tuple[str, ...] = ("memory", "wishbone", "obi")


def wrap_uglir_design(
    design: UGLIRDesign,
    wrap: str | None = None,
    protocol: str | None = None,
    *,
    component_library: dict[str, dict[str, object]] | None = None,
) -> UGLIRDesign:
    """Apply one optional wrapper/protocol composition to one µglIR design."""
    if design.stage != "uglir":
        raise ValueError(f"µglIR wrapping expects µglIR input, got stage '{design.stage}'")
    validate_uglir_for_rtl(design)
    normalized_wrap = _normalize_optional_choice("wrap", wrap, GLUE_WRAPS)
    normalized_protocol = _normalize_optional_protocol(protocol)
    if (normalized_wrap is None) != (normalized_protocol is None):
        raise ValueError("µglIR wrapping requires --wrap and --protocol together")
    if normalized_wrap is None:
        return to_uglir_design(design)
    if normalized_wrap == "none" and normalized_protocol.base == "memory" and not normalized_protocol.features:
        return to_uglir_design(design)
    if normalized_wrap == "slave" and normalized_protocol.base == "wishbone":
        wrapper_plan = plan_slave_wrapper(design, normalized_protocol.base, component_library=component_library)
        protocol_plan = plan_wishbone_slave_protocol(wrapper_plan, features=normalized_protocol.features)
        wrapped = to_uglir_design(
            build_wishbone_slave_wrapper_uglir(
                design,
                wrapper_plan,
                protocol_plan,
                component_library=component_library,
            )
        )
        validate_uglir_for_rtl(wrapped)
        return wrapped
    if normalized_wrap == "slave" and normalized_protocol.base == "obi":
        wrapper_plan = plan_slave_wrapper(design, normalized_protocol.base, component_library=component_library)
        protocol_plan = plan_obi_slave_protocol(wrapper_plan, features=normalized_protocol.features)
        wrapped = to_uglir_design(
            build_obi_slave_wrapper_uglir(
                design,
                wrapper_plan,
                protocol_plan,
                component_library=component_library,
            )
        )
        validate_uglir_for_rtl(wrapped)
        return wrapped
    if normalized_wrap == "master":
        plan_master_wrapper(design, normalized_protocol.base)
        raise NotImplementedError(
            f"µglIR wrapping for wrap='{normalized_wrap}' protocol='{_format_protocol_spec(normalized_protocol)}' is not implemented yet"
        )
    raise NotImplementedError(
        f"µglIR wrapping for wrap='{normalized_wrap}' protocol='{_format_protocol_spec(normalized_protocol)}' is not implemented yet"
    )


def _normalize_optional_choice(name: str, value: str | None, choices: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"unsupported {name} '{value}'")
    return normalized


def _normalize_optional_protocol(value: str | None):
    if value is None:
        return None
    return parse_protocol_spec(value)


def _format_protocol_spec(spec) -> str:
    if not spec.features:
        return spec.base
    return spec.base + "+" + "+".join(spec.features)
