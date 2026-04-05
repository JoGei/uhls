"""Textual µhIR model and parser infrastructure."""

from typing import TYPE_CHECKING

from .model import (
    UHIRConstant,
    UHIRDesign,
    UHIREdge,
    UHIRMux,
    UHIRNode,
    UHIRPort,
    UHIRRegion,
    UHIRRegionRef,
    UHIRResource,
    UHIRSchedule,
    UHIRSourceMap,
    UHIRValueBinding,
)
from .dot import to_dot
from .pretty import format_uhir
from .text import UHIRParseError, parse_uhir, parse_uhir_file

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
    "UHIRConstant",
    "UHIRDesign",
    "UHIREdge",
    "UHIRMux",
    "UHIRNode",
    "UHIRParseError",
    "UHIRPort",
    "UHIRRegion",
    "UHIRRegionRef",
    "UHIRResource",
    "UHIRSchedule",
    "UHIRSourceMap",
    "UHIRValueBinding",
    "build_sequencing_graph",
    "dummy_executability_graph",
    "executability_graph_from_uhir",
    "format_uhir",
    "lower_alloc_to_sched",
    "lower_module_to_seq",
    "lower_sched_to_bind",
    "lower_seq_to_alloc",
    "to_dot",
    "parse_uhir",
    "parse_uhir_file",
]
