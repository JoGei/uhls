"""Generic OBI protocol planning shared across RTL backends."""

from __future__ import annotations

from ..wrap import SlaveWrapperPlan


def plan_obi_slave_protocol(wrapper: SlaveWrapperPlan) -> None:
    """Build the generic OBI-slave protocol plan for one slave wrapper."""
    raise NotImplementedError("generic OBI slave protocol planning is not implemented yet")
