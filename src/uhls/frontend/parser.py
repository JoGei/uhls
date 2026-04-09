"""Recursive-descent parser for the small µC language."""

from __future__ import annotations

from dataclasses import dataclass

from . import ast
from .lexer import Token, tokenize

_TYPE_KEYWORDS = {
    "bool",
    "int8_t",
    "int16_t",
    "int32_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
}

_UNARY_OPS = {"-", "~", "!"}
_BINARY_PRECEDENCE = [
    {"|"},
    {"^"},
    {"&"},
    {"==", "!="},
    {"<", "<=", ">", ">="},
    {"<<", ">>"},
    {"+", "-"},
    {"*", "/", "%"},
]


class ParseError(ValueError):
    """Raised when the source does not match the supported grammar."""


@dataclass
class Parser:
    """Token-stream parser."""

    tokens: list[Token]
    index: int = 0

    def parse_program(self) -> ast.Program:
        functions: list[ast.FunctionDef] = []
        while self.peek().kind != "EOF":
            functions.append(self.parse_function())
        return ast.Program(functions)

    def parse_function(self) -> ast.FunctionDef:
        return_type = self.parse_type()
        name = self.expect("IDENT").text
        self.expect("(")
        params: list[ast.Param] = []
        if self.peek().kind != ")":
            if self.peek().kind == "void" and self.peek(1).kind == ")":
                self.advance()
            else:
                while True:
                    param_type = self.parse_type()
                    param_name = self.expect("IDENT").text
                    if self.match("["):
                        size = int(self.expect("INT").text, 0)
                        self.expect("]")
                        param_type = ast.TypeRef(param_type.name, size)
                    params.append(ast.Param(param_name, param_type))
                    if not self.match(","):
                        break
        self.expect(")")
        body = self.parse_block()
        return ast.FunctionDef(name, return_type, params, body)

    def parse_block(self) -> ast.Block:
        self.expect("{")
        statements: list[ast.Stmt] = []
        while self.peek().kind != "}":
            statements.append(self.parse_statement())
        self.expect("}")
        return ast.Block(statements)

    def parse_statement(self) -> ast.Stmt:
        token = self.peek()
        if token.kind in _TYPE_KEYWORDS:
            statement = self.parse_declaration()
            self.expect(";")
            return statement
        if token.kind == "if":
            return self.parse_if()
        if token.kind == "for":
            return self.parse_for()
        if token.kind == "return":
            self.advance()
            value = self.parse_expression()
            self.expect(";")
            return ast.Return(value)
        if token.kind == "{":
            return self.parse_block()

        if token.kind == "IDENT" and self.peek(1).kind in {"=", "["}:
            statement = self.parse_assignment()
            self.expect(";")
            return statement

        expr = self.parse_expression()
        self.expect(";")
        return ast.ExprStmt(expr)

    def parse_declaration(self) -> ast.Decl:
        type_ref = self.parse_type()
        name = self.expect("IDENT").text
        if self.match("["):
            size = int(self.expect("INT").text, 0)
            self.expect("]")
            type_ref = ast.TypeRef(type_ref.name, size)
        init = None
        if self.match("="):
            if type_ref.array_size is not None and self.peek().kind == "{":
                init = self.parse_array_literal()
            else:
                init = self.parse_expression()
        return ast.Decl(name, type_ref, init)

    def parse_assignment(self) -> ast.Assign | ast.ArrayAssign:
        name = self.expect("IDENT").text
        if self.match("++"):
            return ast.Assign(name, ast.BinaryExpr("+", ast.VarRef(name), ast.IntegerLiteral(1)))
        if self.match("--"):
            return ast.Assign(name, ast.BinaryExpr("-", ast.VarRef(name), ast.IntegerLiteral(1)))
        if self.match("["):
            index = self.parse_expression()
            self.expect("]")
            self.expect("=")
            value = self.parse_expression()
            return ast.ArrayAssign(name, index, value)
        self.expect("=")
        value = self.parse_expression()
        return ast.Assign(name, value)

    def parse_if(self) -> ast.If:
        self.expect("if")
        self.expect("(")
        condition = self.parse_expression()
        self.expect(")")
        then_branch = self.parse_block()
        else_branch = self.parse_block() if self.match("else") else None
        return ast.If(condition, then_branch, else_branch)

    def parse_for(self) -> ast.For:
        self.expect("for")
        self.expect("(")
        init = self.parse_assignment()
        self.expect(";")
        condition = self.parse_expression()
        self.expect(";")
        step = self.parse_assignment()
        self.expect(")")
        body = self.parse_block()
        return ast.For(init, condition, step, body)

    def parse_expression(self) -> ast.Expr:
        return self.parse_binary_level(0)

    def parse_binary_level(self, level: int) -> ast.Expr:
        if level == len(_BINARY_PRECEDENCE):
            return self.parse_unary()

        expr = self.parse_binary_level(level + 1)
        while self.peek().kind in _BINARY_PRECEDENCE[level]:
            op = self.advance().kind
            rhs = self.parse_binary_level(level + 1)
            expr = ast.BinaryExpr(op, expr, rhs)
        return expr

    def parse_unary(self) -> ast.Expr:
        if self.peek().kind in _UNARY_OPS:
            return ast.UnaryExpr(self.advance().kind, self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> ast.Expr:
        token = self.peek()
        if token.kind == "INT":
            return ast.IntegerLiteral(int(self.advance().text, 0))
        if token.kind == "STRING":
            return ast.StringLiteral(self.advance().text)
        if token.kind == "true":
            self.advance()
            return ast.BoolLiteral(True)
        if token.kind == "false":
            self.advance()
            return ast.BoolLiteral(False)
        if token.kind == "IDENT":
            name = self.advance().text
            if self.match("("):
                args: list[ast.Expr] = []
                if self.peek().kind != ")":
                    while True:
                        args.append(self.parse_expression())
                        if not self.match(","):
                            break
                self.expect(")")
                return ast.CallExpr(name, args)
            if self.match("["):
                index = self.parse_expression()
                self.expect("]")
                return ast.ArrayRef(name, index)
            return ast.VarRef(name)
        if token.kind == "(":
            self.advance()
            expr = self.parse_expression()
            self.expect(")")
            return expr
        raise ParseError(f"unexpected token {token.kind!r} at offset {token.position}")

    def parse_array_literal(self) -> ast.ArrayLiteral:
        self.expect("{")
        elements: list[ast.Expr] = []
        if self.peek().kind != "}":
            while True:
                elements.append(self.parse_expression())
                if not self.match(","):
                    break
        self.expect("}")
        return ast.ArrayLiteral(elements)

    def parse_type(self) -> ast.TypeRef:
        while self.match("const"):
            pass
        token = self.peek()
        if token.kind not in _TYPE_KEYWORDS:
            raise ParseError(f"expected type keyword at offset {token.position}")
        self.advance()
        return ast.TypeRef(token.text)

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[self.index + offset]

    def advance(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def expect(self, kind: str) -> Token:
        token = self.peek()
        if token.kind != kind:
            raise ParseError(f"expected {kind!r} at offset {token.position}, got {token.kind!r}")
        return self.advance()

    def match(self, kind: str) -> bool:
        if self.peek().kind != kind:
            return False
        self.advance()
        return True


def parse_program(source: str) -> ast.Program:
    """Parse one full µC translation unit."""
    return Parser(tokenize(source)).parse_program()
