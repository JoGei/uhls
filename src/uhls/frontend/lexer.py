"""Lexer for the small µC language."""

from __future__ import annotations

from dataclasses import dataclass


KEYWORDS = {
    "bool",
    "const",
    "else",
    "false",
    "for",
    "if",
    "int8_t",
    "int16_t",
    "int32_t",
    "return",
    "true",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "void",
}

_TWO_CHAR_TOKENS = {"<=", ">=", "==", "!=", "<<", ">>", "++", "--"}
_ONE_CHAR_TOKENS = set("(){}[];,=+-*/%~!&|^<>")


@dataclass(frozen=True)
class Token:
    """One lexical token."""

    kind: str
    text: str
    position: int


class LexError(ValueError):
    """Raised when the source contains an unsupported token."""


def tokenize(source: str) -> list[Token]:
    """Tokenize one µC source string."""
    tokens: list[Token] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
            continue

        if source.startswith("//", i):
            newline = source.find("\n", i)
            i = len(source) if newline == -1 else newline + 1
            continue

        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end == -1:
                raise LexError("unterminated block comment")
            i = end + 2
            continue

        two = source[i : i + 2]
        if two in _TWO_CHAR_TOKENS:
            tokens.append(Token(two, two, i))
            i += 2
            continue

        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                i += 1
            text = source[start:i]
            kind = text if text in KEYWORDS else "IDENT"
            tokens.append(Token(kind, text, start))
            continue

        if ch.isdigit():
            start = i
            if ch == "0" and i + 1 < len(source) and source[i + 1] in {"x", "X"}:
                i += 2
                hex_start = i
                while i < len(source) and source[i] in "0123456789abcdefABCDEF":
                    i += 1
                if i == hex_start:
                    raise LexError(f"invalid hex literal at offset {start}")
            else:
                i += 1
                while i < len(source) and source[i].isdigit():
                    i += 1
            text = source[start:i]
            tokens.append(Token("INT", text, start))
            continue

        if ch == '"':
            start = i
            i += 1
            value_chars: list[str] = []
            while i < len(source):
                current = source[i]
                if current == '"':
                    i += 1
                    tokens.append(Token("STRING", "".join(value_chars), start))
                    break
                if current == "\\":
                    i += 1
                    if i >= len(source):
                        raise LexError("unterminated string literal")
                    escaped = source[i]
                    escapes = {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}
                    value_chars.append(escapes.get(escaped, escaped))
                    i += 1
                    continue
                value_chars.append(current)
                i += 1
            else:
                raise LexError("unterminated string literal")
            continue

        if ch in _ONE_CHAR_TOKENS:
            tokens.append(Token(ch, ch, i))
            i += 1
            continue

        raise LexError(f"unexpected character {ch!r} at offset {i}")

    tokens.append(Token("EOF", "", len(source)))
    return tokens
