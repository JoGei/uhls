"""Generic RTL protocol planning utilities."""

from __future__ import annotations

from dataclasses import dataclass

from .obi import OBISlaveProtocolPlan, build_obi_slave_wrapper_uglir, plan_obi_slave_protocol
from .wishbone import (
    ProtocolPort,
    WishboneMemoryWindow,
    WishboneScalarRegister,
    WishboneSlaveProtocolPlan,
    build_wishbone_slave_wrapper_uglir,
    plan_wishbone_slave_protocol,
)


@dataclass(frozen=True)
class ProtocolSpec:
    """One parsed protocol specification with optional feature suffixes."""

    base: str
    features: tuple[str, ...] = ()


def parse_protocol_spec(text: str) -> ProtocolSpec:
    """Parse one protocol specification like ``wishbone+err``."""
    normalized = text.strip().lower()
    if not normalized:
        raise ValueError("unsupported protocol ''")
    parts = [part.strip() for part in normalized.split("+")]
    base = parts[0]
    if not base:
        raise ValueError(f"unsupported protocol '{text}'")
    if base not in {"memory", "wishbone", "obi"}:
        raise ValueError(f"unsupported protocol '{text}'")
    features = tuple(part for part in parts[1:] if part)
    if len(features) != len(set(features)):
        raise ValueError(f"protocol '{text}' must not repeat feature suffixes")
    supported = _supported_protocol_features(base)
    unsupported = [feature for feature in features if feature not in supported]
    if unsupported:
        raise ValueError(
            f"unsupported protocol feature(s) for '{base}': {', '.join(unsupported)}"
        )
    return ProtocolSpec(base=base, features=features)


def protocol_spec_help() -> str:
    """Render a short user-facing help string for supported protocol specs."""
    return "memory, wishbone[+err], obi[+burst]"


def _supported_protocol_features(base: str) -> set[str]:
    if base == "wishbone":
        return {"err"}
    if base == "obi":
        return {"burst"}
    return set()

__all__ = [
    "OBISlaveProtocolPlan",
    "ProtocolSpec",
    "ProtocolPort",
    "WishboneMemoryWindow",
    "WishboneScalarRegister",
    "WishboneSlaveProtocolPlan",
    "build_obi_slave_wrapper_uglir",
    "build_wishbone_slave_wrapper_uglir",
    "parse_protocol_spec",
    "plan_obi_slave_protocol",
    "plan_wishbone_slave_protocol",
    "protocol_spec_help",
]
