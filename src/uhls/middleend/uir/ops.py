"""Instruction and terminator objects for canonical µhLS IR."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import ScalarType, TypeLike, normalize_scalar_type
from .values import IncomingValue

COMPACT_OPCODE_LABELS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
    "and": "&",
    "or": "|",
    "xor": "^",
    "shl": "<<",
    "shr": ">>",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "neg": "-",
    "not": "!",
    "mov": "mov",
    "const": "const",
    "load": "ld",
    "store": "st",
    "phi": "phi",
    "call": "call",
    "print": "print",
    "param": "param",
    "br": "br",
    "cbr": "cbr",
    "ret": "ret",
}

_UNARY_OPS = frozenset({"mov", "neg", "not"})
_BINARY_OPS = frozenset({"add", "sub", "mul", "div", "mod", "and", "or", "xor", "shl", "shr"})
_COMPARE_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


@dataclass(slots=True)
class ConstOp:
    """``x:T = const k:T``."""

    dest: str
    type: ScalarType | str
    value: int
    opcode: str = field(default="const", init=False)

    def __post_init__(self) -> None:
        self.type = normalize_scalar_type(self.type)
        self.value = int(self.value)


@dataclass(slots=True)
class UnaryOp:
    """A canonical unary scalar instruction."""

    opcode: str
    dest: str
    type: ScalarType | str
    value: object

    def __post_init__(self) -> None:
        if self.opcode not in _UNARY_OPS:
            raise ValueError(f"unsupported unary opcode '{self.opcode}'")
        self.type = normalize_scalar_type(self.type)


@dataclass(slots=True)
class BinaryOp:
    """A canonical binary scalar instruction."""

    opcode: str
    dest: str
    type: ScalarType | str
    lhs: object
    rhs: object

    def __post_init__(self) -> None:
        if self.opcode not in _BINARY_OPS:
            raise ValueError(f"unsupported binary opcode '{self.opcode}'")
        self.type = normalize_scalar_type(self.type)

    @property
    def operands(self) -> list[object]:
        return [self.lhs, self.rhs]


@dataclass(slots=True)
class CompareOp:
    """A canonical comparison instruction."""

    opcode: str
    dest: str
    lhs: object
    rhs: object
    type: ScalarType | str = field(default="i1")

    def __post_init__(self) -> None:
        if self.opcode not in _COMPARE_OPS:
            raise ValueError(f"unsupported compare opcode '{self.opcode}'")
        self.type = normalize_scalar_type(self.type)

    @property
    def operands(self) -> list[object]:
        return [self.lhs, self.rhs]


@dataclass(slots=True)
class LoadOp:
    """``x:T = load A[idx]``."""

    dest: str
    type: ScalarType | str
    array: str
    index: object
    opcode: str = field(default="load", init=False)

    def __post_init__(self) -> None:
        self.type = normalize_scalar_type(self.type)


@dataclass(slots=True)
class StoreOp:
    """``store A[idx], v``."""

    array: str
    index: object
    value: object
    opcode: str = field(default="store", init=False)


@dataclass(slots=True)
class PhiOp:
    """``x:T = phi(pred1: v1, ..., predN: vN)``."""

    dest: str
    type: ScalarType | str
    incoming: list[IncomingValue | tuple[str, object]]
    opcode: str = field(default="phi", init=False)

    def __post_init__(self) -> None:
        self.type = normalize_scalar_type(self.type)
        normalized: list[IncomingValue] = []
        for item in self.incoming:
            if isinstance(item, IncomingValue):
                normalized.append(item)
            else:
                pred, value = item
                normalized.append(IncomingValue(str(pred), value))
        self.incoming = normalized


@dataclass(slots=True)
class CallOp:
    """One extended direct call instruction."""

    callee: str
    operands: list[object]
    dest: str | None = None
    type: ScalarType | str | None = None
    arg_count: int | None = None
    opcode: str = field(default="call", init=False)

    def __post_init__(self) -> None:
        if self.type is not None:
            self.type = normalize_scalar_type(self.type)
        if self.arg_count is not None:
            self.arg_count = int(self.arg_count)


@dataclass(slots=True)
class PrintOp:
    """One side-effecting formatted print operation."""

    format: str
    operands: list[object]
    opcode: str = field(default="print", init=False)


@dataclass(slots=True)
class ParamOp:
    """One backend-lowered call argument pseudo-op."""

    index: int
    value: object
    opcode: str = field(default="param", init=False)

    @property
    def operands(self) -> list[object]:
        return [self.index, self.value]


@dataclass(slots=True)
class BranchOp:
    """``br target``."""

    target: str
    opcode: str = field(default="br", init=False)


@dataclass(slots=True)
class CondBranchOp:
    """``cbr cond, true_blk, false_blk``."""

    cond: object
    true_target: str
    false_target: str
    opcode: str = field(default="cbr", init=False)


@dataclass(slots=True)
class ReturnOp:
    """``ret v`` or ``ret``."""

    value: object | None = None
    opcode: str = field(default="ret", init=False)


Instruction = (
    ConstOp
    | UnaryOp
    | BinaryOp
    | CompareOp
    | LoadOp
    | StoreOp
    | PhiOp
    | CallOp
    | PrintOp
    | ParamOp
)
Terminator = BranchOp | CondBranchOp | ReturnOp
Operation = Instruction | Terminator
