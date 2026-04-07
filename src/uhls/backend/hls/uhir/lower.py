"""Compatibility wrappers for µhIR lowering stages."""

from __future__ import annotations

from uhls.backend.hls.alloc import lower_seq_to_alloc
from uhls.backend.hls.seq import build_sequencing_graph, lower_module_to_seq

__all__ = ["build_sequencing_graph", "lower_module_to_seq", "lower_seq_to_alloc"]
