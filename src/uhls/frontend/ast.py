"""AST nodes for the µC frontend."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TypeRef:
    """One source-level type reference."""

    name: str
    array_size: int | None = None


@dataclass(frozen=True)
class Param:
    """One function parameter."""

    name: str
    type: TypeRef


@dataclass(frozen=True)
class Expr:
    """Base class for expressions."""


@dataclass(frozen=True)
class IntegerLiteral(Expr):
    """One integer literal."""

    value: int


@dataclass(frozen=True)
class BoolLiteral(Expr):
    """One boolean literal."""

    value: bool


@dataclass(frozen=True)
class StringLiteral(Expr):
    """One string literal."""

    value: str


@dataclass(frozen=True)
class ArrayLiteral(Expr):
    """One brace-delimited array initializer literal."""

    elements: list[Expr]


@dataclass(frozen=True)
class VarRef(Expr):
    """One scalar variable reference."""

    name: str


@dataclass(frozen=True)
class ArrayRef(Expr):
    """One array element read."""

    name: str
    index: Expr


@dataclass(frozen=True)
class CallExpr(Expr):
    """One direct named function call."""

    callee: str
    args: list[Expr]


@dataclass(frozen=True)
class CastExpr(Expr):
    """One explicit scalar cast."""

    type: TypeRef
    value: Expr


@dataclass(frozen=True)
class UnaryExpr(Expr):
    """One unary operation."""

    op: str
    operand: Expr


@dataclass(frozen=True)
class BinaryExpr(Expr):
    """One binary or comparison operation."""

    op: str
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Stmt:
    """Base class for statements."""


@dataclass(frozen=True)
class Block:
    """A compound statement."""

    statements: list[Stmt] = field(default_factory=list)


@dataclass(frozen=True)
class Decl(Stmt):
    """A local declaration, optionally with initialization."""

    name: str
    type: TypeRef
    init: Expr | None = None


@dataclass(frozen=True)
class Assign(Stmt):
    """A scalar assignment."""

    target: str
    value: Expr


@dataclass(frozen=True)
class ArrayAssign(Stmt):
    """An array element assignment."""

    target: str
    index: Expr
    value: Expr


@dataclass(frozen=True)
class If(Stmt):
    """A structured conditional."""

    condition: Expr
    then_branch: Block
    else_branch: Block | None = None


@dataclass(frozen=True)
class For(Stmt):
    """A canonical for loop."""

    init: Assign
    condition: Expr
    step: Assign
    body: Block


@dataclass(frozen=True)
class Return(Stmt):
    """A scalar return statement."""

    value: Expr


@dataclass(frozen=True)
class ExprStmt(Stmt):
    """One expression used as a statement."""

    expr: Expr


@dataclass(frozen=True)
class FunctionDef:
    """One µC function definition."""

    name: str
    return_type: TypeRef
    params: list[Param]
    body: Block


@dataclass(frozen=True)
class Program:
    """A full µC translation unit."""

    functions: list[FunctionDef]
