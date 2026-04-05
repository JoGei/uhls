"""µhIR graph-optimizer pass infrastructure."""

from __future__ import annotations

from copy import deepcopy

from uhls.backend.uhir.model import UHIRDesign
from uhls.middleend.passes.util import PassContext, PassManager

from .registry import (
    GOptPassSpec,
    builtin_gopt_pass_names,
    builtin_gopt_specs,
    create_builtin_gopt_pass,
)


_NON_SEQ_NODE_ATTRIBUTES = frozenset(
    {
        "class",
        "ii",
        "delay",
        "start",
        "end",
        "bind",
        "iter_latency",
        "iter_initiation_interval",
        "iter_ramp_down",
        "slot",
        "stage",
        "guard",
    }
)


def project_to_seq_design(design: UHIRDesign) -> UHIRDesign:
    """Drop alloc/sched/bind artifacts and return one seq-view µhIR design."""
    projected = deepcopy(design)
    projected.stage = "seq"
    projected.schedule = None
    projected.resources = []

    kept_regions = [region for region in projected.regions if region.kind != "executability"]
    kept_region_ids = {region.id for region in kept_regions}
    for region in kept_regions:
        region.region_refs = [ref for ref in region.region_refs if ref.target in kept_region_ids]
        region.value_bindings = []
        region.muxes = []
        region.steps = None
        region.latency = None
        region.initiation_interval = None
        for node in region.nodes:
            node.attributes = {
                name: value for name, value in node.attributes.items() if name not in _NON_SEQ_NODE_ATTRIBUTES
            }
    projected.regions = kept_regions
    return projected


def run_gopt_passes(design: object, pipeline: list[object], *, pass_args: tuple[str, ...] = ()) -> object:
    """Run one µhIR graph-optimizer pipeline."""
    if not isinstance(design, UHIRDesign):
        raise TypeError(f"gopt expects one UHIRDesign, got {type(design).__name__}")
    context = PassContext(pass_args=tuple(pass_args))
    context.data["pass_args"] = list(pass_args)
    return PassManager(list(pipeline)).run(project_to_seq_design(design), context)


__all__ = [
    "GOptPassSpec",
    "builtin_gopt_pass_names",
    "builtin_gopt_specs",
    "create_builtin_gopt_pass",
    "project_to_seq_design",
    "run_gopt_passes",
]
