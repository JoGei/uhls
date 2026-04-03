"""Constant-propagation extension points."""

from __future__ import annotations

from uhls.middleend.uir import Function, Module
from uhls.middleend.passes.util.pass_manager import function_or_module_pass


def const_prop_function(function: Function) -> Function:
    """Optimize one function with constant propagation once implemented by the user."""
    raise NotImplementedError("implement constant propagation here")


def const_prop_module(module: Module) -> Module:
    """Optimize one module with constant propagation once implemented by the user."""
    raise NotImplementedError("implement constant propagation here")


def ConstPropPass():
    """Return a pass-manager compatible constant-propagation pass wrapper."""
    return function_or_module_pass("const_prop", const_prop_function, const_prop_module)
