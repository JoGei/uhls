"""IHP130 SDC helpers."""

from __future__ import annotations


def emit_ihp130_sdc(
    *,
    clock_port: str = "clk",
    clock_name: str = "clk",
    clock_period_ns: float = 10.0,
) -> str:
    """Emit a simple SG13G2-friendly starting SDC."""
    period = _format_period(clock_period_ns)
    return "\n".join(
        [
            f"create_clock [get_ports {{{clock_port}}}] -name {clock_name} -period {period}",
            "",
            "# Add input/output delay constraints once pad timing is defined.",
            "# set_input_delay ...",
            "# set_output_delay ...",
            "",
        ]
    )


def _format_period(value: float) -> str:
    text = f"{value:.3f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


__all__ = ["emit_ihp130_sdc"]
