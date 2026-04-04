"""Hierarchical scheduling composition helpers."""

from __future__ import annotations

# TODO: When sched grows beyond static scheduling, extend this layer to carry
# symbolic or predicate-guarded timing for data-dependent control. In
# particular, runtime-dependent branch choice and non-trivial loop trip counts
# will need to survive here as first-class scheduling information so later FSM
# synthesis can resolve them against datapath-produced values.
