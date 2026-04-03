"""Canonical µhLS IR data model, printer, and verifier."""

from .block import Block
from .function import Function
from .module import Module
from .ops import (
    BinaryOp,
    BranchOp,
    CallOp,
    COMPACT_OPCODE_LABELS,
    CompareOp,
    CondBranchOp,
    ConstOp,
    Instruction,
    LoadOp,
    Operation,
    ParamOp,
    PhiOp,
    PrintOp,
    ReturnOp,
    StoreOp,
    Terminator,
    UnaryOp,
)
from .pretty import format_block, format_function, format_instruction, format_module, format_operand, pretty
from .text import IRParseError, parse_module
from .types import (
    ArrayType,
    SCALAR_TYPE_NAMES,
    ScalarType,
    Type,
    TypeLike,
    coerce_scalar_type,
    normalize_type,
    type_name,
)
from .values import IncomingValue, Literal, Parameter, Variable
from .verifier import IRVerificationError, verify_function, verify_module

__all__ = [
    "ArrayType",
    "BinaryOp",
    "Block",
    "BranchOp",
    "CallOp",
    "COMPACT_OPCODE_LABELS",
    "coerce_scalar_type",
    "CompareOp",
    "CondBranchOp",
    "ConstOp",
    "Function",
    "IRVerificationError",
    "IRParseError",
    "IncomingValue",
    "Instruction",
    "Literal",
    "LoadOp",
    "Module",
    "Operation",
    "ParamOp",
    "Parameter",
    "PhiOp",
    "PrintOp",
    "ReturnOp",
    "SCALAR_TYPE_NAMES",
    "ScalarType",
    "StoreOp",
    "Terminator",
    "Type",
    "TypeLike",
    "UnaryOp",
    "Variable",
    "format_block",
    "format_function",
    "format_instruction",
    "format_module",
    "format_operand",
    "normalize_type",
    "pretty",
    "parse_module",
    "type_name",
    "verify_function",
    "verify_module",
]
