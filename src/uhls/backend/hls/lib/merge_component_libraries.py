"""Merge multiple component-library JSON files into one resolved artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _rewrite_path(value: object, root: Path, out_root: Path) -> object:
    if not isinstance(value, str) or not value.strip() or "$" in value:
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    resolved = (root / path).resolve()
    try:
        return os.path.relpath(resolved, out_root)
    except ValueError:
        return str(resolved)


def merged_component_library_payload(input_paths: list[Path], output_path: Path) -> dict[str, object]:
    out_path = output_path.resolve()
    out_root = out_path.parent
    merged: dict[str, object] = {"components": {}}
    components = merged["components"]
    assert isinstance(components, dict)

    for library_path in input_paths:
        payload = json.loads(library_path.read_text(encoding="utf-8"))
        for name, component in payload.get("components", {}).items():
            copied = json.loads(json.dumps(component))
            hdl = copied.get("hdl")
            if isinstance(hdl, dict):
                if "source" in hdl:
                    hdl["source"] = _rewrite_path(hdl.get("source"), library_path.parent, out_root)
                if isinstance(hdl.get("sources"), list):
                    hdl["sources"] = [_rewrite_path(entry, library_path.parent, out_root) for entry in hdl["sources"]]
                if isinstance(hdl.get("include_dirs"), list):
                    hdl["include_dirs"] = [
                        _rewrite_path(entry, library_path.parent, out_root) for entry in hdl["include_dirs"]
                    ]
            components[str(name)] = copied
    return merged


def merge_component_libraries(input_paths: list[Path], output_path: Path) -> None:
    merged = merged_component_library_payload(input_paths, output_path)
    out_path = output_path.resolve()
    out_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge component-library JSON files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input component-library JSON files")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Merged output JSON path")
    args = parser.parse_args(argv)
    merge_component_libraries([path.resolve() for path in args.inputs], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
