"""Integer semantics shared by the canonical interpreters."""

from __future__ import annotations

from typing import Any

from uhls.middleend.uir import ScalarType, coerce_scalar_type

def parse_int_type(type_hint: Any | None) -> ScalarType | None:
    """Parse a scalar type hint into signedness/width info.

    Args:
        type_hint: A string like ``"i32"`` or an object carrying a similar
            spelling on one of the common IR attributes.

    Returns:
        A :class:`~uhls.middleend.uir.ScalarType` when the hint looks like a supported scalar
        integer type, otherwise ``None``.
    """
    return coerce_scalar_type(type_hint)


def normalize_int(value: int, type_hint: Any | None) -> int:
    """Wrap ``value`` to the bit width and signedness described by ``type_hint``.

    Args:
        value: The integer result that should be canonicalized.
        type_hint: A scalar integer type descriptor such as ``"u8"`` or ``"i32"``.

    Returns:
        The wrapped integer value as interpreted by the requested type.
    """
    info = parse_int_type(type_hint)
    int_value = int(value)
    if info is None:
        return int_value

    modulus = 1 << info.width
    wrapped = int_value % modulus
    if info.signed:
        sign_bit = 1 << (info.width - 1)
        if wrapped & sign_bit:
            return wrapped - modulus
    return wrapped


def int_bits(value: int, type_hint: Any | None) -> int:
    """Return the raw modulo-``2**width`` bit pattern for ``value``.

    Args:
        value: The integer whose bit representation should be masked.
        type_hint: The integer type that provides the target width.

    Returns:
        The unsigned bit-pattern view of ``value`` for the requested width.
    """
    info = parse_int_type(type_hint)
    int_value = int(value)
    if info is None:
        return int_value
    return int_value % (1 << info.width)


def truthy(value: int) -> bool:
    """Interpret an integer as a branch condition.

    Args:
        value: The scalar branch input.

    Returns:
        ``True`` when ``value`` is non-zero, matching µhLS integer truthiness.
    """
    return int(value) != 0


def eval_unary(opcode: str, operand: int, type_hint: Any | None) -> int:
    """Evaluate a canonical unary scalar instruction.

    Args:
        opcode: One of the supported unary opcodes such as ``mov`` or ``neg``.
        operand: The already-resolved integer operand.
        type_hint: The destination/result type used for width normalization.

    Returns:
        The normalized integer result of the unary operation.
    """
    if opcode == "mov":
        result = operand
    elif opcode == "neg":
        result = -operand
    elif opcode == "not":
        result = ~operand
    else:
        raise ValueError(f"unsupported unary opcode '{opcode}'")
    return normalize_int(result, type_hint)


def eval_binary(opcode: str, lhs: int, rhs: int, type_hint: Any | None) -> int:
    """Evaluate a canonical binary scalar instruction.

    Args:
        opcode: One of the supported arithmetic, bitwise, or shift opcodes.
        lhs: The resolved left-hand operand.
        rhs: The resolved right-hand operand.
        type_hint: The destination/result type used for width normalization.

    Returns:
        The normalized integer result of the binary operation.
    """
    shift = max(0, int(rhs))

    if opcode == "add":
        result = lhs + rhs
    elif opcode == "sub":
        result = lhs - rhs
    elif opcode == "mul":
        result = lhs * rhs
    elif opcode == "div":
        result = _trunc_div(lhs, rhs)
    elif opcode == "mod":
        result = lhs - (_trunc_div(lhs, rhs) * rhs)
    elif opcode == "and":
        result = lhs & rhs
    elif opcode == "or":
        result = lhs | rhs
    elif opcode == "xor":
        result = lhs ^ rhs
    elif opcode == "shl":
        result = lhs << shift
    elif opcode == "shr":
        info = parse_int_type(type_hint)
        if info is not None and info.signed:
            result = lhs >> shift
        else:
            result = int_bits(lhs, type_hint) >> shift
    else:
        raise ValueError(f"unsupported binary opcode '{opcode}'")

    return normalize_int(result, type_hint)


def _trunc_div(lhs: int, rhs: int) -> int:
    if rhs == 0:
        raise ZeroDivisionError("integer division by zero")
    quotient = abs(int(lhs)) // abs(int(rhs))
    if (lhs < 0) ^ (rhs < 0):
        return -quotient
    return quotient


def eval_compare(opcode: str, lhs: int, rhs: int) -> int:
    """Evaluate a canonical comparison instruction.

    Args:
        opcode: One of ``eq``, ``ne``, ``lt``, ``le``, ``gt``, or ``ge``.
        lhs: The resolved left-hand operand.
        rhs: The resolved right-hand operand.

    Returns:
        ``1`` when the comparison is true, otherwise ``0``.
    """
    if opcode == "eq":
        result = lhs == rhs
    elif opcode == "ne":
        result = lhs != rhs
    elif opcode == "lt":
        result = lhs < rhs
    elif opcode == "le":
        result = lhs <= rhs
    elif opcode == "gt":
        result = lhs > rhs
    elif opcode == "ge":
        result = lhs >= rhs
    else:
        raise ValueError(f"unsupported compare opcode '{opcode}'")

    return 1 if result else 0
