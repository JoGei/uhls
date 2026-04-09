"""Target-dispatched SDC emission helpers."""

from __future__ import annotations


def emit_sdc(
    target: str | None,
    *,
    clock_port: str = "clk",
    clock_name: str = "clk",
    clock_period_ns: float = 10.0,
) -> str:
    """Emit one target-appropriate SDC skeleton."""
    normalized = _normalize_target(target)
    if normalized == "ihp130":
        from .vendor.ihp130.sdc import emit_ihp130_sdc

        return emit_ihp130_sdc(clock_port=clock_port, clock_name=clock_name, clock_period_ns=clock_period_ns)
    return _emit_generic_sdc(clock_port=clock_port, clock_name=clock_name, clock_period_ns=clock_period_ns)


def _emit_generic_sdc(*, clock_port: str, clock_name: str, clock_period_ns: float) -> str:
    period = _format_period(clock_period_ns)
    return "\n".join(
        [
            f"create_clock [get_ports {{{clock_port}}}] -name {clock_name} -period {period}",
            "",
        ]
    )


def _format_period(value: float) -> str:
    text = f"{value:.3f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _normalize_target(target: str | None) -> str:
    return "" if target is None else target.strip().lower()


__all__ = ["emit_sdc"]
