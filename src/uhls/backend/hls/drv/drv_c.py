"""C driver emission for wrapped µglIR artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from uhls.backend.hls.uglir import UGLIRAddressMap, UGLIRAddressMapEntry, UGLIRDesign


@dataclass(frozen=True)
class _RegisterSpec:
    name: str
    offset: int
    access: str
    symbol: str
    type: str


@dataclass(frozen=True)
class _MemorySpec:
    name: str
    offset: int
    span: int
    access: str
    symbol: str
    word_t: str
    depth: int


def emit_uglir_driver_c(design: UGLIRDesign) -> str:
    """Emit one header-only C HAL for one wrapped µglIR design."""
    if design.stage != "uglir":
        raise ValueError(f"driver emission expects µglIR input, got stage '{design.stage}'")
    address_map = _select_address_map(design)
    registers, memories = _collect_mmio_specs(address_map)
    control_status = next((register for register in registers if register.name == "control_status"), None)
    if control_status is None:
        raise ValueError("wrapped µglIR driver emission requires one 'control_status' register entry")

    scalar_inputs = [register for register in registers if register.name != "control_status" and _is_writable(register.access)]
    scalar_outputs = [register for register in registers if register.name != "control_status" and _is_readable(register.access)]
    result_register = next((register for register in scalar_outputs if register.name == "result"), None)

    design_prefix = _sanitize_c_ident(design.name)
    macro_prefix = f"{design_prefix.upper()}_DRV"
    guard = f"{macro_prefix}_H"
    type_name = f"{design_prefix}_drv_t"
    fn_prefix = f"{design_prefix}_drv"

    lines: list[str] = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdbool.h>",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        f"typedef struct {{",
        "  uintptr_t base_addr;",
        f"}} {type_name};",
        "",
        f"/* {design.name} wrapped MMIO driver",
        f" * protocol: {address_map.name}",
        " */",
        "",
    ]
    lines.extend(_emit_address_macros(macro_prefix, control_status, scalar_inputs, scalar_outputs, memories))
    lines.extend(
        [
            f"enum {{",
            f"  {macro_prefix}_CONTROL_STATUS_DONE_BIT = 0,",
            f"  {macro_prefix}_CONTROL_STATUS_BUSY_BIT = 1,",
            f"  {macro_prefix}_CONTROL_STATUS_START_PENDING_BIT = 2,",
            f"  {macro_prefix}_CONTROL_STATUS_REQ_READY_BIT = 3,",
            "};",
            "",
            f"#define {macro_prefix}_CONTROL_STATUS_DONE_MASK (1u << {macro_prefix}_CONTROL_STATUS_DONE_BIT)",
            f"#define {macro_prefix}_CONTROL_STATUS_BUSY_MASK (1u << {macro_prefix}_CONTROL_STATUS_BUSY_BIT)",
            f"#define {macro_prefix}_CONTROL_STATUS_START_PENDING_MASK (1u << {macro_prefix}_CONTROL_STATUS_START_PENDING_BIT)",
            f"#define {macro_prefix}_CONTROL_STATUS_REQ_READY_MASK (1u << {macro_prefix}_CONTROL_STATUS_REQ_READY_BIT)",
            "",
            f"static inline void {fn_prefix}_init({type_name} *inst, uintptr_t base_addr) {{",
            "  inst->base_addr = base_addr;",
            "}",
            "",
            f"static inline uint32_t {fn_prefix}_mmio_read32(const {type_name} *inst, uintptr_t rel_offset) {{",
            "  return *(volatile const uint32_t *)(inst->base_addr + rel_offset);",
            "}",
            "",
            f"static inline void {fn_prefix}_mmio_write32({type_name} *inst, uintptr_t rel_offset, uint32_t value) {{",
            "  *(volatile uint32_t *)(inst->base_addr + rel_offset) = value;",
            "}",
            "",
            f"static inline uint32_t {fn_prefix}_read_control_status(const {type_name} *inst) {{",
            f"  return {fn_prefix}_mmio_read32(inst, {macro_prefix}_REG_CONTROL_STATUS_OFFSET);",
            "}",
            "",
            f"static inline bool {fn_prefix}_done(const {type_name} *inst) {{",
            f"  return ({fn_prefix}_read_control_status(inst) & {macro_prefix}_CONTROL_STATUS_DONE_MASK) != 0u;",
            "}",
            "",
            f"static inline bool {fn_prefix}_busy(const {type_name} *inst) {{",
            f"  return ({fn_prefix}_read_control_status(inst) & {macro_prefix}_CONTROL_STATUS_BUSY_MASK) != 0u;",
            "}",
            "",
            f"static inline bool {fn_prefix}_start_pending(const {type_name} *inst) {{",
            f"  return ({fn_prefix}_read_control_status(inst) & {macro_prefix}_CONTROL_STATUS_START_PENDING_MASK) != 0u;",
            "}",
            "",
            f"static inline bool {fn_prefix}_req_ready(const {type_name} *inst) {{",
            f"  return ({fn_prefix}_read_control_status(inst) & {macro_prefix}_CONTROL_STATUS_REQ_READY_MASK) != 0u;",
            "}",
            "",
            f"static inline void {fn_prefix}_start_nonblocking({type_name} *inst) {{",
            f"  {fn_prefix}_mmio_write32(inst, {macro_prefix}_REG_CONTROL_STATUS_OFFSET, 1u);",
            "}",
            "",
            f"static inline void {fn_prefix}_wait(const {type_name} *inst) {{",
            f"  while (!{fn_prefix}_done(inst)) {{",
            "  }",
            "}",
            "",
        ]
    )

    for register in registers:
        if register.name == "control_status":
            continue
        lines.extend(_emit_register_accessors(fn_prefix, type_name, macro_prefix, register))

    for memory in memories:
        lines.extend(_emit_memory_accessors(fn_prefix, type_name, macro_prefix, memory))

    if scalar_inputs:
        lines.extend(_emit_configure_function(fn_prefix, type_name, scalar_inputs))
    lines.extend(_emit_start_blocking_function(fn_prefix, type_name, scalar_inputs, result_register))
    lines.append(f"#endif /* {guard} */")
    return "\n".join(lines)


def _emit_address_macros(
    macro_prefix: str,
    control_status: _RegisterSpec,
    scalar_inputs: list[_RegisterSpec],
    scalar_outputs: list[_RegisterSpec],
    memories: list[_MemorySpec],
) -> list[str]:
    lines = _emit_register_macros(macro_prefix, control_status)
    for register in scalar_inputs:
        lines.extend(_emit_register_macros(macro_prefix, register))
    for register in scalar_outputs:
        if any(existing.name == register.name for existing in scalar_inputs):
            continue
        lines.extend(_emit_register_macros(macro_prefix, register))
    for memory in memories:
        lines.extend(_emit_memory_macros(macro_prefix, memory))
    return lines


def _emit_register_macros(macro_prefix: str, register: _RegisterSpec) -> list[str]:
    name = _sanitize_macro_suffix(register.name)
    width_bytes = _type_size_bytes(register.type)
    end = register.offset + max(width_bytes - 1, 0)
    return [
        f"#define {macro_prefix}_REG_{name}_OFFSET ((uintptr_t){_c_hex(register.offset)})",
        f"#define {macro_prefix}_REG_{name}_RANGE_START ((uintptr_t){_c_hex(register.offset)})",
        f"#define {macro_prefix}_REG_{name}_RANGE_END ((uintptr_t){_c_hex(end)})",
        f"#define {macro_prefix}_REG_{name}_SYMBOL \"{register.symbol}\"",
        "",
    ]


def _emit_memory_macros(macro_prefix: str, memory: _MemorySpec) -> list[str]:
    name = _sanitize_macro_suffix(memory.name)
    end = memory.offset + max(memory.span - 1, 0)
    word_bytes = _type_size_bytes(memory.word_t)
    return [
        f"#define {macro_prefix}_MEM_{name}_OFFSET ((uintptr_t){_c_hex(memory.offset)})",
        f"#define {macro_prefix}_MEM_{name}_RANGE_START ((uintptr_t){_c_hex(memory.offset)})",
        f"#define {macro_prefix}_MEM_{name}_RANGE_END ((uintptr_t){_c_hex(end)})",
        f"#define {macro_prefix}_MEM_{name}_DEPTH ((size_t){memory.depth}u)",
        f"#define {macro_prefix}_MEM_{name}_WORD_BYTES ((size_t){word_bytes}u)",
        f"#define {macro_prefix}_MEM_{name}_SYMBOL \"{memory.symbol}\"",
        "",
    ]


def _emit_register_accessors(fn_prefix: str, type_name: str, macro_prefix: str, register: _RegisterSpec) -> list[str]:
    name = _sanitize_c_ident(register.name)
    macro_name = _sanitize_macro_suffix(register.name)
    c_type = _c_type_for_uglir_type(register.type)
    lines: list[str] = []
    if _is_readable(register.access):
        lines.extend(
            [
                f"static inline {c_type} {fn_prefix}_get_{name}(const {type_name} *inst) {{",
                f"  return ({c_type}){fn_prefix}_mmio_read32(inst, {macro_prefix}_REG_{macro_name}_OFFSET);",
                "}",
                "",
            ]
        )
    if _is_writable(register.access):
        lines.extend(
            [
                f"static inline void {fn_prefix}_set_{name}({type_name} *inst, {c_type} value) {{",
                f"  {fn_prefix}_mmio_write32(inst, {macro_prefix}_REG_{macro_name}_OFFSET, (uint32_t)value);",
                "}",
                "",
            ]
        )
    return lines


def _emit_memory_accessors(fn_prefix: str, type_name: str, macro_prefix: str, memory: _MemorySpec) -> list[str]:
    name = _sanitize_c_ident(memory.name)
    macro_name = _sanitize_macro_suffix(memory.name)
    c_type = _c_type_for_uglir_type(memory.word_t)
    lines: list[str] = [
        f"static inline uintptr_t {fn_prefix}_{name}_addr(size_t index) {{",
        f"  return {macro_prefix}_MEM_{macro_name}_OFFSET + ((uintptr_t)index * {macro_prefix}_MEM_{macro_name}_WORD_BYTES);",
        "}",
        "",
    ]
    if _is_readable(memory.access):
        lines.extend(
            [
                f"static inline {c_type} {fn_prefix}_read_{name}(const {type_name} *inst, size_t index) {{",
                f"  return ({c_type}){fn_prefix}_mmio_read32(inst, {fn_prefix}_{name}_addr(index));",
                "}",
                "",
            ]
        )
    if _is_writable(memory.access):
        lines.extend(
            [
                f"static inline void {fn_prefix}_write_{name}({type_name} *inst, size_t index, {c_type} value) {{",
                f"  {fn_prefix}_mmio_write32(inst, {fn_prefix}_{name}_addr(index), (uint32_t)value);",
                "}",
                "",
            ]
        )
    return lines


def _emit_configure_function(fn_prefix: str, type_name: str, scalar_inputs: list[_RegisterSpec]) -> list[str]:
    params = ", ".join(f"{_c_type_for_uglir_type(register.type)} value_{_sanitize_c_ident(register.name)}" for register in scalar_inputs)
    lines = [
        f"static inline void {fn_prefix}_configure({type_name} *inst, {params}) {{",
    ]
    for register in scalar_inputs:
        safe_name = _sanitize_c_ident(register.name)
        lines.append(f"  {fn_prefix}_set_{safe_name}(inst, value_{safe_name});")
    lines.extend(["}", ""])
    return lines


def _emit_start_blocking_function(
    fn_prefix: str,
    type_name: str,
    scalar_inputs: list[_RegisterSpec],
    result_register: _RegisterSpec | None,
) -> list[str]:
    params = ", ".join(f"{_c_type_for_uglir_type(register.type)} value_{_sanitize_c_ident(register.name)}" for register in scalar_inputs)
    signature_tail = f", {params}" if params else ""
    return_type = _c_type_for_uglir_type(result_register.type) if result_register is not None else "void"
    lines = [
        f"static inline {return_type} {fn_prefix}_start_blocking({type_name} *inst{signature_tail}) {{",
    ]
    if scalar_inputs:
        call_args = ", ".join(f"value_{_sanitize_c_ident(register.name)}" for register in scalar_inputs)
        lines.append(f"  {fn_prefix}_configure(inst, {call_args});")
    lines.append(f"  {fn_prefix}_start_nonblocking(inst);")
    lines.append(f"  {fn_prefix}_wait(inst);")
    if result_register is not None:
        lines.append(f"  return {fn_prefix}_get_{_sanitize_c_ident(result_register.name)}(inst);")
    lines.extend(["}", ""])
    return lines


def _select_address_map(design: UGLIRDesign) -> UGLIRAddressMap:
    if not design.address_maps:
        raise ValueError("driver emission requires wrapped µglIR with one address_map block")
    return design.address_maps[0]


def _collect_mmio_specs(address_map: UGLIRAddressMap) -> tuple[list[_RegisterSpec], list[_MemorySpec]]:
    registers: list[_RegisterSpec] = []
    memories: list[_MemorySpec] = []
    for entry in address_map.entries:
        if entry.kind == "register":
            registers.append(_register_spec(entry))
        elif entry.kind == "memory":
            memories.append(_memory_spec(entry))
    return registers, memories


def _register_spec(entry: UGLIRAddressMapEntry) -> _RegisterSpec:
    attrs = entry.attributes
    return _RegisterSpec(
        name=entry.name,
        offset=_parse_u32_like(attrs.get("offset")),
        access=str(attrs.get("access", "rw")),
        symbol=str(attrs.get("symbol", entry.name)),
        type=str(attrs.get("type", "u32")),
    )


def _memory_spec(entry: UGLIRAddressMapEntry) -> _MemorySpec:
    attrs = entry.attributes
    return _MemorySpec(
        name=entry.name,
        offset=_parse_u32_like(attrs.get("offset")),
        span=_parse_u32_like(attrs.get("span")),
        access=str(attrs.get("access", "rw")),
        symbol=str(attrs.get("symbol", entry.name)),
        word_t=str(attrs.get("word_t", "u32")),
        depth=int(attrs.get("depth", 0)),
    )


def _parse_u32_like(value: object) -> int:
    text = str(value).strip().lower()
    match = re.fullmatch(r"32'h([0-9a-f]{4})_([0-9a-f]{4})", text)
    if match is None:
        raise ValueError(f"unsupported u32 literal '{value}'")
    return (int(match.group(1), 16) << 16) | int(match.group(2), 16)


def _type_size_bytes(type_hint: str) -> int:
    match = re.fullmatch(r"[iu](\d+)", type_hint)
    if match is None:
        raise ValueError(f"unsupported driver-visible type '{type_hint}'")
    width = int(match.group(1))
    return max((width + 7) // 8, 1)


def _c_type_for_uglir_type(type_hint: str) -> str:
    if type_hint == "i1":
        return "bool"
    match = re.fullmatch(r"([iu])(\d+)", type_hint)
    if match is None:
        raise ValueError(f"unsupported C driver type '{type_hint}'")
    signed, width_text = match.groups()
    width = int(width_text)
    if width not in {8, 16, 32, 64}:
        raise ValueError(f"unsupported C driver width '{type_hint}'")
    prefix = "int" if signed == "i" else "uint"
    return f"{prefix}{width}_t"


def _is_writable(access: str) -> bool:
    return access.startswith("rw")


def _is_readable(access: str) -> bool:
    return access.startswith("rw") or access.startswith("ro")


def _sanitize_c_ident(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not sanitized:
        sanitized = "unnamed"
    if sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    return sanitized.lower()


def _sanitize_macro_suffix(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not sanitized:
        sanitized = "UNNAMED"
    if sanitized[0].isdigit():
        sanitized = f"N_{sanitized}"
    return sanitized.upper()


def _c_hex(value: int) -> str:
    return f"0x{value:x}u"


__all__ = ["emit_uglir_driver_c"]
