"""Textual µhIR model and parser infrastructure."""

from typing import TYPE_CHECKING

from .model import (
    UHIRAssign,
    UHIRAttach,
    UHIRConstant,
    UHIRController,
    UHIRControllerEmit,
    UHIRControllerLink,
    UHIRControllerState,
    UHIRControllerTransition,
    UHIRDesign,
    UHIREdge,
    UHIRGlueMux,
    UHIRGlueMuxCase,
    UHIRMux,
    UHIRNode,
    UHIRPort,
    UHIRRegion,
    UHIRRegionRef,
    UHIRResource,
    UHIRSchedule,
    UHIRSeqBlock,
    UHIRSeqUpdate,
    UHIRSourceMap,
    UHIRValueBinding,
)
from .dot import to_dot
from .gopt import (
    GOptPassSpec,
    builtin_gopt_pass_names,
    builtin_gopt_specs,
    create_builtin_gopt_pass,
    project_to_seq_design,
    run_gopt_passes,
)
from .pretty import format_uhir
from .text import UHIRParseError, parse_uhir, parse_uhir_file
from .timing import TimingBinary, TimingCall, TimingExpr, TimingUnary, TimingVar, parse_timing_expr

if TYPE_CHECKING:
    from uhls.backend.hls.alloc import ExecutabilityGraph


def build_sequencing_graph(*args: object, **kwargs: object) -> object:
    """Lower canonical µIR to the internal sequencing-graph dialect."""
    from uhls.backend.hls.seq import build_sequencing_graph as _impl

    return _impl(*args, **kwargs)


def lower_module_to_seq(*args: object, **kwargs: object) -> object:
    """Lower canonical µIR to seq-stage µhIR."""
    from uhls.backend.hls.seq import lower_module_to_seq as _impl

    return _impl(*args, **kwargs)


def lower_seq_to_alloc(*args: object, **kwargs: object) -> object:
    """Lower seq-stage µhIR to alloc-stage µhIR."""
    from uhls.backend.hls.alloc import lower_seq_to_alloc as _impl

    return _impl(*args, **kwargs)


def lower_alloc_to_sched(*args: object, **kwargs: object) -> object:
    """Lower alloc-stage µhIR to sched-stage µhIR."""
    from uhls.backend.hls.sched import lower_alloc_to_sched as _impl

    return _impl(*args, **kwargs)


def lower_sched_to_bind(*args: object, **kwargs: object) -> object:
    """Lower sched-stage µhIR to bind-stage µhIR."""
    from uhls.backend.hls.bind import lower_sched_to_bind as _impl

    return _impl(*args, **kwargs)


def lower_bind_to_fsm(*args: object, **kwargs: object) -> object:
    """Lower bind-stage µhIR to fsm-stage µhIR."""
    from uhls.backend.hls.fsm import lower_bind_to_fsm as _impl

    return _impl(*args, **kwargs)


def lower_fsm_to_uglir(*args: object, **kwargs: object) -> object:
    """Lower fsm-stage µhIR to uglir-stage µhIR."""
    from uhls.backend.hls.uglir import lower_fsm_to_uglir as _impl

    return _impl(*args, **kwargs)


def dummy_executability_graph(*args: object, **kwargs: object) -> object:
    """Build one starter executability graph covering canonical µIR ops."""
    from uhls.backend.hls.alloc import dummy_executability_graph as _impl

    return _impl(*args, **kwargs)


def executability_graph_from_uhir(*args: object, **kwargs: object) -> object:
    """Build one alloc executability graph from exg-stage µhIR."""
    from uhls.backend.hls.alloc import executability_graph_from_uhir as _impl

    return _impl(*args, **kwargs)


def __getattr__(name: str) -> object:
    if name == "ExecutabilityGraph":
        from uhls.backend.hls.alloc import ExecutabilityGraph as _impl

        return _impl
    raise AttributeError(name)

__all__ = [
    "ExecutabilityGraph",
    "GOptPassSpec",
    "UHIRAssign",
    "UHIRAttach",
    "UHIRConstant",
    "UHIRController",
    "UHIRControllerEmit",
    "UHIRControllerLink",
    "UHIRControllerState",
    "UHIRControllerTransition",
    "UHIRDesign",
    "UHIREdge",
    "UHIRGlueMux",
    "UHIRGlueMuxCase",
    "UHIRMux",
    "UHIRNode",
    "UHIRParseError",
    "UHIRPort",
    "UHIRRegion",
    "UHIRRegionRef",
    "UHIRResource",
    "UHIRSchedule",
    "UHIRSeqBlock",
    "UHIRSeqUpdate",
    "UHIRSourceMap",
    "UHIRValueBinding",
    "TimingBinary",
    "TimingCall",
    "TimingExpr",
    "TimingUnary",
    "TimingVar",
    "builtin_gopt_pass_names",
    "builtin_gopt_specs",
    "create_builtin_gopt_pass",
    "project_to_seq_design",
    "build_sequencing_graph",
    "dummy_executability_graph",
    "executability_graph_from_uhir",
    "format_uhir",
    "lower_alloc_to_sched",
    "lower_bind_to_fsm",
    "lower_fsm_to_uglir",
    "lower_module_to_seq",
    "lower_sched_to_bind",
    "lower_seq_to_alloc",
    "to_dot",
    "parse_uhir",
    "parse_uhir_file",
    "parse_timing_expr",
    "run_gopt_passes",
]
