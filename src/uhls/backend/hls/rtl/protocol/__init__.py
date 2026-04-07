"""Generic RTL protocol planning utilities."""

from .obi import plan_obi_slave_protocol
from .wishbone import (
    ProtocolPort,
    WishboneMemoryWindow,
    WishboneScalarRegister,
    WishboneSlaveProtocolPlan,
    plan_wishbone_slave_protocol,
)

__all__ = [
    "ProtocolPort",
    "WishboneMemoryWindow",
    "WishboneScalarRegister",
    "WishboneSlaveProtocolPlan",
    "plan_obi_slave_protocol",
    "plan_wishbone_slave_protocol",
]
