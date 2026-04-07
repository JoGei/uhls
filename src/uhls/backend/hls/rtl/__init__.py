"""Generic RTL emission backends."""

from .lower import RTL_HDLS, RTL_RESETS, ResetSpec, lower_uglir_to_rtl, parse_reset_spec
from .protocol import parse_protocol_spec, plan_obi_slave_protocol, plan_wishbone_slave_protocol, protocol_spec_help
from .verilog import emit_uglir_to_verilog
from .wrap import plan_master_wrapper, plan_slave_wrapper

__all__ = [
    "RTL_HDLS",
    "RTL_RESETS",
    "ResetSpec",
    "emit_uglir_to_verilog",
    "lower_uglir_to_rtl",
    "parse_reset_spec",
    "parse_protocol_spec",
    "plan_obi_slave_protocol",
    "plan_master_wrapper",
    "plan_slave_wrapper",
    "plan_wishbone_slave_protocol",
    "protocol_spec_help",
]
