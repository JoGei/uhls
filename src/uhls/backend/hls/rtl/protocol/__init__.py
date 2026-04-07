"""Generic RTL protocol planning utilities."""

from .obi import plan_obi_slave_protocol
from .wishbone import (
    ProtocolPort,
    WishboneMemoryWindow,
    WishboneScalarRegister,
    WishboneSlaveProtocolPlan,
    build_wishbone_slave_wrapper_uglir,
    plan_wishbone_slave_protocol,
)

__all__ = [
    "ProtocolPort",
    "WishboneMemoryWindow",
    "WishboneScalarRegister",
    "WishboneSlaveProtocolPlan",
    "build_wishbone_slave_wrapper_uglir",
    "plan_obi_slave_protocol",
    "plan_wishbone_slave_protocol",
]
