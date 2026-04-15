"""Transformation and optimization passes for canonical µhLS IR."""

from uhls.middleend.passes.opt.canonicalize_loops import (
    CanonicalizeLoopsPass,
    canonicalize_loops_function,
    canonicalize_loops_module,
)
from uhls.middleend.passes.opt.const_prop import ConstPropPass, const_prop_function, const_prop_module
from uhls.middleend.passes.opt.copy_prop import CopyPropPass, copy_prop_function, copy_prop_module
from uhls.middleend.passes.opt.cse import CSEPass, cse_function, cse_module
from uhls.middleend.passes.opt.dce import DCEPass, dce_function, dce_module
from uhls.middleend.passes.opt.inline_calls import InlineCallsPass, InlineError, inline_calls
from uhls.middleend.passes.opt.mov_to_add_zero import (
    MovToAddZeroPass,
    mov_to_add_zero_function,
    mov_to_add_zero_module,
)
from uhls.middleend.passes.opt.prune_functions import (
    PruneFunctionsPass,
    prune_functions_function,
    prune_functions_module,
)
from uhls.middleend.passes.opt.simplify_cfg import (
    SimplifyCFGPass,
    simplify_cfg_function,
    simplify_cfg_module,
)
from uhls.middleend.passes.opt.unroll_loops import (
    UnrollLoopsPass,
    unroll_loops_function,
    unroll_loops_module,
)

__all__ = [
    "CanonicalizeLoopsPass",
    "CSEPass",
    "ConstPropPass",
    "CopyPropPass",
    "DCEPass",
    "InlineCallsPass",
    "InlineError",
    "MovToAddZeroPass",
    "PruneFunctionsPass",
    "SimplifyCFGPass",
    "UnrollLoopsPass",
    "canonicalize_loops_function",
    "canonicalize_loops_module",
    "const_prop_function",
    "const_prop_module",
    "copy_prop_function",
    "copy_prop_module",
    "cse_function",
    "cse_module",
    "dce_function",
    "dce_module",
    "inline_calls",
    "mov_to_add_zero_function",
    "mov_to_add_zero_module",
    "prune_functions_function",
    "prune_functions_module",
    "simplify_cfg_function",
    "simplify_cfg_module",
    "unroll_loops_function",
    "unroll_loops_module",
]
