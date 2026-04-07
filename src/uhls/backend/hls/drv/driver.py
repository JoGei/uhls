"""Language-dispatched driver generation for wrapped µglIR artifacts."""

from __future__ import annotations

from .drv_c import emit_uglir_driver_c

DRV_LANGS: tuple[str, ...] = ("c",)


def emit_uglir_driver(design, *, lang: str) -> str:
    """Emit one software-driver artifact for one wrapped µglIR design."""
    normalized = lang.strip().lower()
    if normalized == "c":
        return emit_uglir_driver_c(design)
    raise NotImplementedError(f"driver target language '{lang}' is not implemented yet")


__all__ = ["DRV_LANGS", "emit_uglir_driver"]
