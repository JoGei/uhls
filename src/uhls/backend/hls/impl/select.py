"""Implementation-selection helpers used during allocation refinement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from uhls.backend.hls.lib import format_component_spec


MEM_POLICIES = ("ffonly", "autoram")


@dataclass(slots=True, frozen=True)
class MemoryPolicy:
    """One memory implementation-selection policy."""

    mode: str = "ffonly"
    threshold_bits: int = 1024


@dataclass(slots=True, frozen=True)
class MemoryImplementationChoice:
    """One refined memory implementation choice."""

    component_spec: str
    load_ii: int
    load_delay: int
    store_ii: int
    store_delay: int


@dataclass(slots=True, frozen=True)
class _VendorMemoryMacro:
    component_name: str
    width_bits: int
    depth_words: int
    load_ii: int = 1
    load_delay: int = 2
    store_ii: int = 1
    store_delay: int = 1


def parse_memory_policy(spec: str | None) -> MemoryPolicy:
    """Parse one alloc --mem specification."""
    if spec is None or not spec.strip():
        return MemoryPolicy()
    text = spec.strip().lower()
    if text == "ffonly":
        return MemoryPolicy(mode="ffonly")
    if text.startswith("autoram"):
        threshold_bits = 1024
        suffix = text[len("autoram") :]
        if suffix:
            if not suffix.startswith("+"):
                raise ValueError("autoram memory policy must use syntax autoram[+<bits>] or autoram[+<bits>bits]")
            threshold_text = suffix[1:]
            if threshold_text.endswith("bits"):
                threshold_text = threshold_text[:-4]
            if not threshold_text.isdigit() or int(threshold_text) <= 0:
                raise ValueError("autoram memory policy threshold must be one positive integer bit count")
            threshold_bits = int(threshold_text)
        return MemoryPolicy(mode="autoram", threshold_bits=threshold_bits)
    raise ValueError("memory policy must be one of: ffonly, autoram[+<bits>]")


def select_memory_implementation(
    *,
    component_name: str,
    element_type: str,
    depth_words: int | None,
    policy: MemoryPolicy,
    vendor_components: Mapping[str, Mapping[str, object]] | None,
) -> MemoryImplementationChoice:
    """Select one concrete memory implementation contract."""
    generic_spec = _generic_memory_spec(component_name, element_type, depth_words)
    generic_choice = MemoryImplementationChoice(
        component_spec=generic_spec,
        load_ii=1,
        load_delay=1,
        store_ii=1,
        store_delay=1,
    )
    if policy.mode == "ffonly":
        return generic_choice

    if vendor_components is None:
        raise ValueError("autoram memory policy requires one vendor component library")

    total_bits = _type_width_bits(element_type) * (1 if depth_words is None else depth_words)
    if total_bits < policy.threshold_bits:
        return generic_choice

    macros = _vendor_memory_macros(vendor_components)
    width_bits = _type_width_bits(element_type)
    width_matches = [macro for macro in macros if macro.width_bits == width_bits]
    if not width_matches:
        return generic_choice
    if depth_words is None:
        return generic_choice

    fitting = [macro for macro in width_matches if depth_words <= macro.depth_words]
    if not fitting:
        raise ValueError(
            f"autoram cannot support memory word_t={element_type} word_len={depth_words}: "
            "vendor component library has no single macro fit and banked memories are not supported yet"
        )
    chosen = min(
        fitting,
        key=lambda macro: (macro.depth_words * macro.width_bits, macro.depth_words, macro.component_name),
    )
    component_spec = format_component_spec(
        chosen.component_name,
        {
            "word_t": element_type,
            "word_len": str(depth_words),
        },
    )
    return MemoryImplementationChoice(
        component_spec=component_spec,
        load_ii=chosen.load_ii,
        load_delay=chosen.load_delay,
        store_ii=chosen.store_ii,
        store_delay=chosen.store_delay,
    )


def _generic_memory_spec(component_name: str, element_type: str, depth_words: int | None) -> str:
    params = {"word_t": element_type}
    if depth_words is not None:
        params["word_len"] = str(depth_words)
    return format_component_spec(component_name, params)


def _type_width_bits(type_name: str) -> int:
    if len(type_name) < 2 or type_name[0] not in {"i", "u"} or not type_name[1:].isdigit():
        raise ValueError(f"memory implementation selection requires one integer word type, got '{type_name}'")
    return int(type_name[1:])


def _vendor_memory_macros(
    vendor_components: Mapping[str, Mapping[str, object]],
) -> tuple[_VendorMemoryMacro, ...]:
    macros: list[_VendorMemoryMacro] = []
    for component_name, component in vendor_components.items():
        if component.get("kind") != "memory":
            continue
        memory_shape = component.get("memory")
        if not isinstance(memory_shape, Mapping):
            continue
        word_t = memory_shape.get("word_t")
        word_len = memory_shape.get("word_len")
        if not isinstance(word_t, str) or not isinstance(word_len, int) or word_len <= 0:
            continue
        supports = component.get("supports")
        if not isinstance(supports, Mapping):
            continue
        load_support = supports.get("load")
        store_support = supports.get("store")
        if not isinstance(load_support, Mapping) or not isinstance(store_support, Mapping):
            continue
        load_ii = load_support.get("ii")
        load_delay = load_support.get("d")
        store_ii = store_support.get("ii")
        store_delay = store_support.get("d")
        if not all(isinstance(value, int) and value >= 1 for value in (load_ii, load_delay, store_ii, store_delay)):
            continue
        macros.append(
            _VendorMemoryMacro(
                component_name=component_name,
                width_bits=_type_width_bits(word_t),
                depth_words=word_len,
                load_ii=int(load_ii),
                load_delay=int(load_delay),
                store_ii=int(store_ii),
                store_delay=int(store_delay),
            )
        )
    return tuple(macros)
