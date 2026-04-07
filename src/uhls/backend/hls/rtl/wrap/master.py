"""Generic master-wrapper planning shared across RTL backends."""

from __future__ import annotations

from uhls.backend.hls.uhir.model import UHIRDesign


def plan_master_wrapper(design: UHIRDesign, protocol: str) -> None:
    """Build one generic master-wrapper plan for one uglir core."""
    raise NotImplementedError(
        f"generic master wrapper planning for protocol '{protocol.strip().lower()}' is not implemented yet"
    )
