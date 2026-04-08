#!/usr/bin/env python3
"""Import IHP SG13G2 SRAM macros into one uhls component library JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from uhls.backend.hls.lib import import_verilog_component_stub_from_files, validate_component_library


_TOP_MODULE_RE = re.compile(r"^RM_IHPSG13_(?P<ports>[12]P)_(?P<depth>\d+)x(?P<width>\d+)_.*$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path, help="Existing uhls component library JSON")
    parser.add_argument(
        "--env",
        dest="env_mappings",
        action="append",
        default=[],
        help="Preserve one path prefix as NAME=/absolute/path and rewrite generated HDL paths to ${NAME}/...",
    )
    parser.add_argument(
        "--verilog-dir",
        type=Path,
        required=True,
        help="Path to ihp-sg13g2 SRAM verilog directory",
    )
    parser.add_argument(
        "--include-2p",
        action="store_true",
        help="Also import 2-port SRAM macros. Default imports only 1-port macros.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path. Defaults to overwriting the input library.",
    )
    args = parser.parse_args()

    library_path: Path = args.library
    verilog_dir: Path = args.verilog_dir
    output_path: Path = library_path if args.output is None else args.output
    preserved_envs = _parse_env_mappings(tuple(args.env_mappings))

    payload = _load_or_initialize_library_json(library_path)
    if not isinstance(payload, dict):
        raise SystemExit(f"component library '{library_path}' must be one JSON object")
    components = payload.get("components")
    if components is None:
        components = {}
        payload["components"] = components
    if not isinstance(components, dict):
        raise SystemExit(f"component library '{library_path}' must define object-valued 'components'")

    source_paths = tuple(sorted(verilog_dir.glob("*.v")))
    source_files = tuple((path, path.read_text(encoding="utf-8", errors="ignore")) for path in source_paths)
    imported_count = 0
    for module_name in _discover_ihp_sram_modules(source_files, include_2p=args.include_2p):
        if module_name in components:
            continue
        imported = import_verilog_component_stub_from_files(
            source_files=source_files,
            module_name=module_name,
            kind="memory",
        )
        _rewrite_hdl_paths_with_env(imported, preserved_envs)
        _enrich_ihp_memory_stub(imported, module_name)
        components[module_name] = imported
        imported_count += 1

    for component_payload in components.values():
        if isinstance(component_payload, dict):
            _rewrite_hdl_paths_with_env(component_payload, preserved_envs)

    payload["components"] = validate_component_library(components)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"imported {imported_count} IHP SRAM macro(s) into {output_path}")
    return 0


def _discover_ihp_sram_modules(
    source_files: tuple[tuple[Path, str], ...],
    *,
    include_2p: bool,
) -> list[str]:
    modules: list[str] = []
    seen: set[str] = set()
    for _path, text in source_files:
        for module_name in re.findall(r"\bmodule\s+([A-Za-z_][\w$]*)\b", text):
            match = _TOP_MODULE_RE.fullmatch(module_name)
            if match is None:
                continue
            if match.group("ports") == "2P" and not include_2p:
                continue
            if module_name not in seen:
                seen.add(module_name)
                modules.append(module_name)
    return sorted(modules)


def _enrich_ihp_memory_stub(component: dict[str, object], module_name: str) -> None:
    match = _TOP_MODULE_RE.fullmatch(module_name)
    if match is None:
        return
    port_kind = match.group("ports")
    depth = int(match.group("depth"))
    width = int(match.group("width"))
    hdl = component.get("hdl")
    if isinstance(hdl, dict):
        hdl.setdefault("defines", ["FUNCTIONAL"])
    ports = component.get("ports")
    if isinstance(ports, dict):
        _annotate_ihp_memory_ports(ports)
    component["parameters"] = {
        "word_t": {"kind": "type", "required": True},
        "word_len": {"kind": "int", "required": True},
    }
    component["memory"] = {
        "word_t": f"i{width}",
        "word_len": depth,
    }
    if port_kind == "1P":
        component["supports"] = {
            "load": {
                "ii": 1,
                "d": 2,
                "mode": "read",
                "bind": {
                    "A_DLY": "true",
                    "A_MEN": "true",
                    "A_REN": "true",
                    "A_WEN": "false",
                    "A_ADDR": "operand1",
                    "A_DOUT": "result",
                },
            },
            "store": {
                "ii": 1,
                "d": 1,
                "mode": "write",
                "bind": {
                    "A_DLY": "true",
                    "A_MEN": "true",
                    "A_REN": "false",
                    "A_WEN": "true",
                    "A_ADDR": "operand1",
                    "A_DIN": "operand2",
                },
            },
        }
        if isinstance(ports, dict) and "A_BM" in ports:
            component["supports"]["store"]["bind"]["A_BM"] = _full_mask_literal(width)


def _annotate_ihp_memory_ports(ports: dict[str, object]) -> None:
    for port_name, port in ports.items():
        if not isinstance(port, dict):
            continue
        if port_name in {"A_CLK", "B_CLK"}:
            port["type"] = "clock"
            continue
        if port_name in {
            "A_DIN",
            "A_DOUT",
            "A_BM",
            "A_BIST_DIN",
            "A_BIST_BM",
            "B_DIN",
            "B_DOUT",
            "B_BM",
            "B_BIST_DIN",
            "B_BIST_BM",
        }:
            port["type"] = "word_t"
        if "_BIST_" in port_name and port.get("dir") == "input":
            port["tie"] = "false"


def _full_mask_literal(width: int) -> str:
    return f"{(1 << width) - 1}:u{width}"


def _expand_env_in_json(value: object) -> object:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [_expand_env_in_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_in_json(item) for key, item in value.items()}
    return value


def _load_or_initialize_library_json(library_path: Path) -> object:
    try:
        text = library_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"components": {}}
    except OSError as exc:
        raise SystemExit(f"failed to read component library '{library_path}': {exc}") from exc
    if not text.strip():
        return {"components": {}}
    try:
        return _expand_env_in_json(json.loads(text))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse component library '{library_path}': {exc}") from exc


_ENV_REF_RE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_(]))"
)


def _expand_env_string(text: str) -> str:
    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain")
        assert name is not None
        value = os.environ.get(name)
        if value is None:
            missing.append(name)
            return match.group(0)
        return value

    expanded = _ENV_REF_RE.sub(_replace, text)
    if missing:
        missing_text = ", ".join(sorted(dict.fromkeys(missing)))
        raise ValueError(f"undefined environment variable(s): {missing_text}")
    return expanded


def _parse_env_mappings(items: tuple[str, ...]) -> dict[str, Path]:
    mappings: dict[str, Path] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if sep != "=" or not name or not value:
            raise SystemExit(f"invalid --env mapping '{item}'; expected NAME=/absolute/path")
        path = Path(value).resolve()
        mappings[name] = path
    return mappings


def _rewrite_hdl_paths_with_env(component: dict[str, object], env_mappings: dict[str, Path]) -> None:
    if not env_mappings:
        return
    hdl = component.get("hdl")
    if not isinstance(hdl, dict):
        return
    for field in ("source",):
        value = hdl.get(field)
        if isinstance(value, str):
            hdl[field] = _rewrite_path_with_env(value, env_mappings)
    for field in ("sources", "include_dirs"):
        values = hdl.get(field)
        if isinstance(values, list):
            hdl[field] = [
                _rewrite_path_with_env(value, env_mappings) if isinstance(value, str) else value
                for value in values
            ]


def _rewrite_path_with_env(path_text: str, env_mappings: dict[str, Path]) -> str:
    path = Path(path_text).resolve()
    best_name: str | None = None
    best_root: Path | None = None
    for name, root in env_mappings.items():
        root = root.resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if best_root is None or len(root.parts) > len(best_root.parts):
            best_name = name
            best_root = root
            best_relative = relative
    if best_root is None or best_name is None:
        return path_text
    if str(best_relative) == ".":
        return f"${{{best_name}}}"
    return f"${{{best_name}}}/{best_relative.as_posix()}"


if __name__ == "__main__":
    raise SystemExit(main())
