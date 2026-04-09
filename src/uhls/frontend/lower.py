"""Lower the µC AST into canonical µIR."""

from __future__ import annotations

from dataclasses import dataclass, field

from uhls.middleend.uir import (
    ArrayType,
    BinaryOp,
    Block,
    BranchOp,
    CallOp,
    CompareOp,
    CondBranchOp,
    ConstOp,
    Function,
    Literal,
    LoadOp,
    Module,
    Parameter,
    PrintOp,
    ReturnOp,
    StoreOp,
    UnaryOp,
    Variable,
    type_name,
)

from . import ast
from .parser import parse_program
from .sema import FunctionInfo, ProgramInfo, analyze_program, lowerable_type
from .ssa import to_uir_function

_UNARY_OP_MAP = {"-": "neg", "~": "not", "!": "not"}
_BINARY_OP_MAP = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "mod",
    "&": "and",
    "|": "or",
    "^": "xor",
    "<<": "shl",
    ">>": "shr",
}
_COMPARE_OP_MAP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}


@dataclass
class _BlockBuilder:
    label: str
    instructions: list[object] = field(default_factory=list)
    terminator: object | None = None

    def build(self) -> Block:
        return Block(self.label, list(self.instructions), self.terminator)


@dataclass
class _FunctionLowerer:
    function: ast.FunctionDef
    info: FunctionInfo
    temp_counter: int = 0
    block_counter: int = 0
    blocks: list[_BlockBuilder] = field(default_factory=list)
    local_array_symbols: dict[str, str] = field(default_factory=dict)
    local_arrays: dict[str, dict[str, object]] = field(default_factory=dict)

    def lower(self) -> Function:
        self.collect_local_arrays(self.function.body)
        entry = self.new_block("entry")
        current = self.lower_block(self.function.body, entry)
        if current is not None and current.terminator is None:
            raise ValueError(f"function '{self.function.name}' does not end in return")

        params = [
            Parameter(param.name, lowerable_type(param.type))
            for param in self.function.params
        ]
        return Function(
            name=self.function.name,
            params=params,
            blocks=[block.build() for block in self.blocks],
            return_type=lowerable_type(self.function.return_type),
            entry="entry",
            local_arrays=dict(self.local_arrays),
        )

    def lower_block(self, block: ast.Block, current: _BlockBuilder | None) -> _BlockBuilder | None:
        active = current
        for statement in block.statements:
            if active is None:
                break
            active = self.lower_statement(statement, active)
        return active

    def lower_statement(self, statement: ast.Stmt, current: _BlockBuilder) -> _BlockBuilder | None:
        if isinstance(statement, ast.Block):
            return self.lower_block(statement, current)

        if isinstance(statement, ast.Decl):
            if statement.type.array_size is None and statement.init is not None:
                value = self.lower_expr(statement.init, current)
                self.assign_scalar(statement.name, value, self.info.expr_types[id(statement.init)], current)
                return current
            if statement.type.array_size is not None and isinstance(statement.init, ast.ArrayLiteral):
                array_name = self.resolve_array_name(statement.name)
                for index, element in enumerate(statement.init.elements):
                    value = self.lower_expr(element, current)
                    current.instructions.append(StoreOp(array_name, Literal(index, "i32"), value))
            return current

        if isinstance(statement, ast.Assign):
            value = self.lower_expr(statement.value, current)
            self.assign_scalar(statement.target, value, self.info.symbols[statement.target], current)
            return current

        if isinstance(statement, ast.ArrayAssign):
            index = self.lower_expr(statement.index, current)
            value = self.lower_expr(statement.value, current)
            current.instructions.append(StoreOp(self.resolve_array_name(statement.target), index, value))
            return current

        if isinstance(statement, ast.Return):
            value = self.lower_expr(statement.value, current)
            current.terminator = ReturnOp(value)
            return None

        if isinstance(statement, ast.ExprStmt):
            if isinstance(statement.expr, ast.CallExpr) and statement.expr.callee == "uhls_printf":
                self.lower_printf(statement.expr, current)
                return current
            self.lower_expr(statement.expr, current)
            return current

        if isinstance(statement, ast.If):
            condition = self.lower_expr(statement.condition, current)
            then_block = self.new_block("if_then")
            else_block = self.new_block("if_else")
            merge_block = self.new_block("if_end")
            current.terminator = CondBranchOp(condition, then_block.label, else_block.label)

            then_end = self.lower_block(statement.then_branch, then_block)
            if then_end is not None and then_end.terminator is None:
                then_end.terminator = BranchOp(merge_block.label)

            else_branch = statement.else_branch or ast.Block()
            else_end = self.lower_block(else_branch, else_block)
            if else_end is not None and else_end.terminator is None:
                else_end.terminator = BranchOp(merge_block.label)

            if then_end is None and else_end is None:
                return None
            return merge_block

        if isinstance(statement, ast.For):
            self.lower_statement(statement.init, current)
            header = self.new_block("for_header")
            body = self.new_block("for_body")
            latch = self.new_block("for_latch")
            exit_block = self.new_block("for_exit")
            current.terminator = BranchOp(header.label)

            condition = self.lower_expr(statement.condition, header)
            header.terminator = CondBranchOp(condition, body.label, exit_block.label)

            body_end = self.lower_block(statement.body, body)
            if body_end is not None and body_end.terminator is None:
                body_end.terminator = BranchOp(latch.label)

            if latch.terminator is None:
                self.lower_statement(statement.step, latch)
                latch.terminator = BranchOp(header.label)
            return exit_block

        raise ValueError(f"unsupported statement {statement!r}")

    def lower_expr(self, expr: ast.Expr, current: _BlockBuilder) -> object:
        if isinstance(expr, ast.IntegerLiteral):
            return Literal(expr.value, self.info.expr_types[id(expr)])
        if isinstance(expr, ast.BoolLiteral):
            return Literal(1 if expr.value else 0, "i1")
        if isinstance(expr, ast.VarRef):
            return Variable(expr.name, self.info.expr_types[id(expr)])
        if isinstance(expr, ast.ArrayRef):
            index = self.lower_expr(expr.index, current)
            target_type = self.info.expr_types[id(expr)]
            temp = self.new_temp()
            current.instructions.append(LoadOp(temp, target_type, self.resolve_array_name(expr.name), index))
            return Variable(temp, target_type)
        if isinstance(expr, ast.CallExpr):
            operands: list[object] = []
            signature = self.info.signatures[expr.callee]
            for arg, param_type in zip(expr.args, signature.param_types, strict=True):
                if isinstance(param_type, ArrayType):
                    assert isinstance(arg, ast.VarRef)
                    operands.append(Variable(self.resolve_array_name(arg.name), param_type))
                    continue
                operands.append(self.coerce_value(self.lower_expr(arg, current), param_type, current))
            result_type = self.info.expr_types[id(expr)]
            temp = self.new_temp()
            current.instructions.append(CallOp(expr.callee, operands, dest=temp, type=result_type))
            return Variable(temp, result_type)
        if isinstance(expr, ast.CastExpr):
            value = self.lower_expr(expr.value, current)
            result_type = self.info.expr_types[id(expr)]
            return self.coerce_value(value, result_type, current)
        if isinstance(expr, ast.UnaryExpr):
            operand = self.lower_expr(expr.operand, current)
            result_type = self.info.expr_types[id(expr)]
            temp = self.new_temp()
            opcode = _UNARY_OP_MAP[expr.op]
            current.instructions.append(UnaryOp(opcode, temp, result_type, operand))
            return Variable(temp, result_type)
        if isinstance(expr, ast.BinaryExpr):
            lhs = self.lower_expr(expr.lhs, current)
            rhs = self.lower_expr(expr.rhs, current)
            result_type = self.info.expr_types[id(expr)]
            temp = self.new_temp()
            if expr.op in _COMPARE_OP_MAP:
                lhs_type = self.info.expr_types[id(expr.lhs)]
                rhs_type = self.info.expr_types[id(expr.rhs)]
                if type_name(lhs_type) != type_name(rhs_type):
                    rhs = self.coerce_value(rhs, lhs_type, current)
                current.instructions.append(CompareOp(_COMPARE_OP_MAP[expr.op], temp, lhs, rhs))
            else:
                if expr.op not in {"<<", ">>"}:
                    lhs = self.coerce_value(lhs, result_type, current)
                    rhs = self.coerce_value(rhs, result_type, current)
                current.instructions.append(BinaryOp(_BINARY_OP_MAP[expr.op], temp, result_type, lhs, rhs))
            return Variable(temp, result_type)
        raise ValueError(f"unsupported expression {expr!r}")

    def assign_scalar(self, target: str, value: object, target_type: object, current: _BlockBuilder) -> None:
        if isinstance(value, Literal):
            current.instructions.append(ConstOp(target, target_type, value.value))
            return
        if isinstance(value, Variable) and value.name == target:
            return
        current.instructions.append(UnaryOp("mov", target, target_type, value))

    def coerce_value(self, value: object, target_type: object, current: _BlockBuilder) -> object:
        if type_name(getattr(value, "type", None)) == type_name(target_type):
            return value
        if isinstance(value, Literal):
            return Literal(value.value, target_type)
        temp = self.new_temp()
        current.instructions.append(UnaryOp("mov", temp, target_type, value))
        return Variable(temp, target_type)

    def collect_local_arrays(self, block: ast.Block) -> None:
        for statement in block.statements:
            if isinstance(statement, ast.Block):
                self.collect_local_arrays(statement)
                continue
            if isinstance(statement, ast.If):
                self.collect_local_arrays(statement.then_branch)
                if statement.else_branch is not None:
                    self.collect_local_arrays(statement.else_branch)
                continue
            if isinstance(statement, ast.For):
                self.collect_local_arrays(statement.body)
                continue
            if not isinstance(statement, ast.Decl) or statement.type.array_size is None:
                continue
            lowered_type = lowerable_type(statement.type)
            assert isinstance(lowered_type, ArrayType)
            lowered_name = f"{self.function.name}${statement.name}"
            self.local_array_symbols[statement.name] = lowered_name
            self.local_arrays[lowered_name] = {
                "size": statement.type.array_size,
                "element_type": lowered_type.element_type,
            }

    def resolve_array_name(self, name: str) -> str:
        return self.local_array_symbols.get(name, name)

    def lower_printf(self, expr: ast.CallExpr, current: _BlockBuilder) -> None:
        assert isinstance(expr.args[0], ast.StringLiteral)
        operands = [self.lower_expr(arg, current) for arg in expr.args[1:]]
        current.instructions.append(PrintOp(expr.args[0].value, operands))

    def new_temp(self) -> str:
        name = f"t{self.temp_counter}"
        self.temp_counter += 1
        return name

    def new_block(self, prefix: str) -> _BlockBuilder:
        label = prefix if prefix == "entry" and not self.blocks else f"{prefix}_{self.block_counter}"
        self.block_counter += 1
        block = _BlockBuilder(label)
        self.blocks.append(block)
        return block


def _lower_function_plain(function: ast.FunctionDef, info: ProgramInfo | None = None) -> Function:
    """Lower one µC function to pre-SSA frontend IR."""
    program = ast.Program([function])
    program_info = info or analyze_program(program)
    return _FunctionLowerer(function, program_info.functions[function.name]).lower()


def lower_program(program: ast.Program, info: ProgramInfo | None = None) -> Module:
    """Lower one full µC translation unit to canonical µIR."""
    program_info = info or analyze_program(program)
    return Module(functions=[lower_function(function, program_info) for function in program.functions])


def lower_function(function: ast.FunctionDef, info: ProgramInfo | None = None) -> Function:
    """Lower one µC function to canonical µIR."""
    return to_uir_function(_lower_function_plain(function, info))


def lower_source_to_uir(source: str) -> Module:
    """Parse, analyze, and lower one µC source string."""
    program = parse_program(source)
    return lower_program(program)
