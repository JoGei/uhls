"""Semantic checks for the small µC frontend."""

from __future__ import annotations

from dataclasses import dataclass, field

from uhls.middleend.uir import ArrayType, ScalarType, type_name
from uhls.utils.graph import assert_acyclic

from . import ast

_TYPE_MAP = {
    "bool": ScalarType("i1"),
    "int8_t": ScalarType("i8"),
    "int16_t": ScalarType("i16"),
    "int32_t": ScalarType("i32"),
    "uint8_t": ScalarType("u8"),
    "uint16_t": ScalarType("u16"),
    "uint32_t": ScalarType("u32"),
}
_ARITHMETIC_OPS = {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}
_COMPARE_OPS = {"<", "<=", ">", ">=", "==", "!="}


class SemanticError(ValueError):
    """Raised when a µC program violates the supported subset."""


@dataclass
class FunctionInfo:
    """Typed information derived for one function."""

    symbols: dict[str, object] = field(default_factory=dict)
    expr_types: dict[int, ScalarType] = field(default_factory=dict)
    local_arrays: dict[str, ArrayType] = field(default_factory=dict)
    called_functions: set[str] = field(default_factory=set)
    return_type: ScalarType | None = None
    signatures: dict[str, "FunctionSignature"] = field(default_factory=dict)


@dataclass(frozen=True)
class FunctionSignature:
    """Callable signature information for one source function."""

    return_type: ScalarType
    param_types: tuple[object, ...]


@dataclass
class ProgramInfo:
    """Typed information derived for a full program."""

    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    signatures: dict[str, FunctionSignature] = field(default_factory=dict)

    def expr_type(self, expr: ast.Expr, function_name: str) -> ScalarType:
        """Return the inferred scalar type of ``expr`` in ``function_name``."""
        return self.functions[function_name].expr_types[id(expr)]


def lowerable_type(type_ref: ast.TypeRef) -> ScalarType | ArrayType:
    """Map one µC source type into canonical IR type objects."""
    try:
        scalar = _TYPE_MAP[type_ref.name]
    except KeyError as exc:
        raise SemanticError(f"unsupported source type '{type_ref.name}'") from exc
    if type_ref.array_size is None:
        return scalar
    if type_ref.array_size <= 0:
        raise SemanticError("array sizes must be positive")
    return ArrayType(scalar)


def analyze_program(program: ast.Program) -> ProgramInfo:
    """Analyze all functions in one program."""
    info = ProgramInfo()
    seen: set[str] = set()
    for function in program.functions:
        if function.name in seen:
            raise SemanticError(f"duplicate function '{function.name}'")
        seen.add(function.name)
        info.signatures[function.name] = _function_signature(function)
    for function in program.functions:
        info.functions[function.name] = analyze_function(function, info.signatures)
    _reject_recursive_calls(info)
    return info


def analyze_function(
    function: ast.FunctionDef,
    signatures: dict[str, FunctionSignature] | None = None,
) -> FunctionInfo:
    """Analyze one function definition."""
    signature = signatures[function.name] if signatures is not None else _function_signature(function)
    info = FunctionInfo(return_type=signature.return_type, signatures=signatures or {})
    for param in function.params:
        _bind_symbol(info, param.name, lowerable_type(param.type))

    _check_block(function.body, function.name, info)
    return info


def _check_block(block: ast.Block, function_name: str, info: FunctionInfo) -> None:
    for statement in block.statements:
        _check_statement(statement, function_name, info)


def _check_statement(statement: ast.Stmt, function_name: str, info: FunctionInfo) -> None:
    if isinstance(statement, ast.Block):
        _check_block(statement, function_name, info)
        return

    if isinstance(statement, ast.Decl):
        declared_type = lowerable_type(statement.type)
        _bind_symbol(info, statement.name, declared_type)
        if isinstance(declared_type, ArrayType):
            info.local_arrays[statement.name] = declared_type
            if statement.init is not None:
                if not isinstance(statement.init, ast.ArrayLiteral):
                    raise SemanticError(f"array declaration '{statement.name}' requires a brace initializer")
                assert statement.type.array_size is not None
                if len(statement.init.elements) > statement.type.array_size:
                    raise SemanticError(
                        f"array initializer for '{statement.name}' has {len(statement.init.elements)} elements, "
                        f"but the declared size is {statement.type.array_size}"
                    )
                for element in statement.init.elements:
                    _check_assignment_type(declared_type.element_type, element, function_name, info)
            return
        if statement.init is not None:
            _check_assignment_type(declared_type, statement.init, function_name, info)
        return

    if isinstance(statement, ast.Assign):
        target_type = _require_scalar_symbol(info, statement.target)
        _check_assignment_type(target_type, statement.value, function_name, info)
        return

    if isinstance(statement, ast.ArrayAssign):
        array_type = _require_array_symbol(info, statement.target)
        _check_scalar_expr(statement.index, function_name, info)
        _check_assignment_type(array_type.element_type, statement.value, function_name, info)
        return

    if isinstance(statement, ast.If):
        _check_scalar_expr(statement.condition, function_name, info)
        _check_block(statement.then_branch, function_name, info)
        if statement.else_branch is not None:
            _check_block(statement.else_branch, function_name, info)
        return

    if isinstance(statement, ast.For):
        if not isinstance(statement.init, ast.Assign) or not isinstance(statement.step, ast.Assign):
            raise SemanticError("for-loop init and step must be simple assignments")
        _require_scalar_symbol(info, statement.init.target)
        _require_scalar_symbol(info, statement.step.target)
        _check_statement(statement.init, function_name, info)
        _check_scalar_expr(statement.condition, function_name, info)
        _check_statement(statement.step, function_name, info)
        _check_block(statement.body, function_name, info)
        return

    if isinstance(statement, ast.Return):
        assert info.return_type is not None
        _check_assignment_type(info.return_type, statement.value, function_name, info)
        return

    if isinstance(statement, ast.ExprStmt):
        if isinstance(statement.expr, ast.CallExpr) and statement.expr.callee == "uhls_printf":
            _check_uhls_printf(statement.expr, function_name, info)
            return
        _check_scalar_expr(statement.expr, function_name, info)
        return

    raise SemanticError(f"unsupported statement node {statement!r}")


def _check_assignment_type(
    target_type: ScalarType,
    value: ast.Expr,
    function_name: str,
    info: FunctionInfo,
) -> None:
    expr_type = _check_scalar_expr(value, function_name, info, expected_type=target_type)
    if type_name(expr_type) == "i1" and type_name(target_type) != "i1":
        return
    if type_name(target_type) == "i1" and type_name(expr_type) != "i1":
        return


def _check_scalar_expr(
    expr: ast.Expr,
    function_name: str,
    info: FunctionInfo,
    expected_type: ScalarType | None = None,
) -> ScalarType:
    cached = info.expr_types.get(id(expr))
    if cached is not None:
        if expected_type is not None and isinstance(expr, ast.IntegerLiteral):
            info.expr_types[id(expr)] = expected_type
            return expected_type
        return cached

    if isinstance(expr, ast.IntegerLiteral):
        result = expected_type or ScalarType("i32")
    elif isinstance(expr, ast.BoolLiteral):
        result = ScalarType("i1")
    elif isinstance(expr, ast.VarRef):
        result = _require_scalar_symbol(info, expr.name)
    elif isinstance(expr, ast.ArrayRef):
        array_type = _require_array_symbol(info, expr.name)
        _check_scalar_expr(expr.index, function_name, info)
        result = array_type.element_type
    elif isinstance(expr, ast.CallExpr):
        if expr.callee == "uhls_printf":
            raise SemanticError("uhls_printf may only be used as a statement")
        try:
            signature = info.signatures[expr.callee]
        except KeyError as exc:
            raise SemanticError(f"call to undeclared function '{expr.callee}'") from exc

        if len(expr.args) != len(signature.param_types):
            raise SemanticError(
                f"call to '{expr.callee}' expects {len(signature.param_types)} arguments, got {len(expr.args)}"
            )
        for arg, param_type in zip(expr.args, signature.param_types, strict=True):
            if isinstance(param_type, ArrayType):
                if not isinstance(arg, ast.VarRef):
                    raise SemanticError(f"array parameter to '{expr.callee}' must be passed by name")
                _require_array_symbol(info, arg.name)
                continue
            _check_scalar_expr(arg, function_name, info, expected_type=param_type)
        info.called_functions.add(expr.callee)
        result = signature.return_type
    elif isinstance(expr, ast.StringLiteral | ast.ArrayLiteral):
        raise SemanticError(f"unsupported expression node {expr!r}")
    elif isinstance(expr, ast.UnaryExpr):
        operand_type = _check_scalar_expr(
            expr.operand,
            function_name,
            info,
            expected_type=None if expr.op == "!" else expected_type,
        )
        result = ScalarType("i1") if expr.op == "!" else operand_type
    elif isinstance(expr, ast.BinaryExpr):
        if expr.op in _COMPARE_OPS:
            lhs_type = _check_scalar_expr(expr.lhs, function_name, info)
            rhs_type = _check_scalar_expr(expr.rhs, function_name, info, expected_type=lhs_type)
            if type_name(lhs_type) != type_name(rhs_type) and isinstance(expr.lhs, ast.IntegerLiteral):
                lhs_type = _check_scalar_expr(expr.lhs, function_name, info, expected_type=rhs_type)
            result = ScalarType("i1")
        elif expr.op in _ARITHMETIC_OPS:
            operand_type = expected_type if expected_type is not None and type_name(expected_type) != "i1" else None
            lhs_type = _check_scalar_expr(expr.lhs, function_name, info, expected_type=operand_type)
            rhs_type = _check_scalar_expr(
                expr.rhs,
                function_name,
                info,
                expected_type=operand_type or lhs_type,
            )
            if type_name(lhs_type) != type_name(rhs_type) and isinstance(expr.lhs, ast.IntegerLiteral):
                lhs_type = _check_scalar_expr(
                    expr.lhs,
                    function_name,
                    info,
                    expected_type=rhs_type,
                )
            result = operand_type or lhs_type
        else:
            raise SemanticError(f"unsupported binary operator '{expr.op}'")
    else:
        raise SemanticError(f"unsupported expression node {expr!r}")

    info.expr_types[id(expr)] = result
    return result


def _bind_symbol(info: FunctionInfo, name: str, symbol_type: object) -> None:
    if name in info.symbols:
        raise SemanticError(f"duplicate declaration of '{name}'")
    info.symbols[name] = symbol_type


def _require_scalar_symbol(info: FunctionInfo, name: str) -> ScalarType:
    try:
        symbol_type = info.symbols[name]
    except KeyError as exc:
        raise SemanticError(f"use of undeclared symbol '{name}'") from exc
    if isinstance(symbol_type, ArrayType):
        raise SemanticError(f"symbol '{name}' is an array, not a scalar")
    return symbol_type


def _require_array_symbol(info: FunctionInfo, name: str) -> ArrayType:
    try:
        symbol_type = info.symbols[name]
    except KeyError as exc:
        raise SemanticError(f"use of undeclared symbol '{name}'") from exc
    if not isinstance(symbol_type, ArrayType):
        raise SemanticError(f"symbol '{name}' is not an array")
    return symbol_type


def _check_uhls_printf(expr: ast.CallExpr, function_name: str, info: FunctionInfo) -> None:
    if not expr.args:
        raise SemanticError("uhls_printf requires a format string")
    if not isinstance(expr.args[0], ast.StringLiteral):
        raise SemanticError("uhls_printf requires a string literal as its first argument")
    for arg in expr.args[1:]:
        _check_scalar_expr(arg, function_name, info)


def _function_signature(function: ast.FunctionDef) -> FunctionSignature:
    return_type = lowerable_type(function.return_type)
    if isinstance(return_type, ArrayType):
        raise SemanticError(f"function '{function.name}' cannot return an array type")
    return FunctionSignature(
        return_type=return_type,
        param_types=tuple(lowerable_type(param.type) for param in function.params),
    )


def _reject_recursive_calls(program_info: ProgramInfo) -> None:
    edges = {
        function_name: {
            callee for callee in info.called_functions if callee in program_info.functions
        }
        for function_name, info in program_info.functions.items()
    }
    assert_acyclic(
        edges,
        lambda function_name: edges[function_name],
        cycle_error=lambda function_name: SemanticError(
            f"recursive cycle involving '{function_name}' is not supported"
        ),
    )
