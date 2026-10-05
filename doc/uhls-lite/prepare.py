#!/usr/bin/env python3
"""Stage a browser copy of the current µIR cookbook; never edit the originals.

Run from a checkout containing src/uhls and doc/notebooks/uir_cookbook.ipynb.
Only this script's _build directory is replaced, with a safety marker check.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

DISTRIBUTION = "uhls-browser-demo"
SETUP_START = "repo_root = Path.cwd().resolve()"
SETUP_IMPORT = "from uhls.frontend import lower_source_to_uir"
SETUP_PRINT = 'print(f"repo_root = {repo_root}")'
MARKER = ".generated-by-uhls-lite"


def source_text(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def code_cell(cell_id: str, text: str) -> dict:
    # Permit top-level await, as used by IPython/Pyodide.
    compile(text, f"<{cell_id}>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def dependencies(repo: Path) -> list[str]:
    """Accept the repository's current plain PEP 508 requirements format."""
    reqs = []
    for line in (repo / "requirements.txt").read_text(encoding="utf-8").splitlines():
        item = line.split(" #", 1)[0].strip()
        if not item or item.startswith("#"):
            continue
        if item.startswith("-") or item.endswith("\\"):
            raise ValueError("requirements.txt now uses pip options/includes; adapt packaging first")
        reqs.append(item)
    return reqs


def fingerprint(package: Path, reqs: list[str]) -> str:
    digest = hashlib.sha256(json.dumps(reqs, sort_keys=True).encode())
    for path in sorted(package.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "0.0.0+src" + digest.hexdigest()[:12]


def adapt_notebook(original: dict, version: str, graph_source: str, text_only: bool) -> dict:
    result = copy.deepcopy(original)
    setup_cells = [c for c in result["cells"] if c.get("id") == "setup-code"]
    if len(setup_cells) != 1:
        raise ValueError("Expected exactly one setup-code cell; adapt prepare.py to the new notebook")
    setup = setup_cells[0]
    old = source_text(setup)
    for marker in (SETUP_START, SETUP_IMPORT, SETUP_PRINT):
        if old.count(marker) != 1:
            raise ValueError(f"Cookbook setup changed: expected one occurrence of {marker!r}")
    if old.index(SETUP_START) >= old.index(SETUP_IMPORT):
        raise ValueError("Cookbook setup ordering changed")
    before = old[:old.index(SETUP_START)]
    after = old[old.index(SETUP_IMPORT):].replace(SETUP_PRINT, 'print("µhLS browser package loaded")')
    installs = [f"{DISTRIBUTION}=={version}"]
    if not text_only:
        try:
            installs.append("anywidget==" + importlib.metadata.version("anywidget"))
        except importlib.metadata.PackageNotFoundError:
            # Allows stdlib-only preparation/tests; real builds install requirements.txt first.
            installs.append("anywidget")
    new_setup = before + "import piplite\nawait piplite.install(" + repr(installs) + ")\n\n" + after
    setup["source"] = new_setup.splitlines(keepends=True)
    imports_replaced = 0
    for cell in result["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = source_text(cell)
        if cell is not setup and "repo_root" in text:
            raise ValueError("A new notebook cell depends on repository files; adapt it before exporting")
        imports_replaced += text.count("from graphviz import Source")
        text = text.replace("from graphviz import Source", "# Source is defined by the browser setup above")
        cell["source"] = text.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
        compile(text, "<adapted-cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    if not imports_replaced:
        raise ValueError("The cookbook's Graphviz import changed; review the adapter")
    helper = ('def Source(dot_source: str):\n    print(dot_source)\n'
              if text_only else graph_source)
    index = result["cells"].index(setup)
    result["cells"].insert(index + 1, code_cell("browser-graphviz", helper))
    for cell in result["cells"]:
        if cell.get("id") == "setup-intro":
            cell["source"] = ["## Browser setup\n\n", "Run this first. µhLS is installed from a wheel bundled with this site.\n",
                              "This is a browser kernel, not the repository's local Python environment.\n"]
    result["metadata"]["kernelspec"] = {"display_name": "Python (Pyodide)", "language": "python", "name": "python"}
    result["metadata"].pop("widgets", None)
    result["metadata"]["language_info"] = {"name": "python"}
    return result


def smoke_notebook(version: str) -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python (Pyodide)", "language": "python", "name": "python"}},
            "cells": [
                {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": [
                    "# µhLS browser smoke test\n\nRun both cells. The expected result is 16, before and after optimization.\n",
                    "This notebook deliberately does not test Graphviz or downstream EDA executables.\n"]},
                code_cell("install", f'import piplite\nawait piplite.install("{DISTRIBUTION}=={version}")\n'),
                code_cell("compiler", '''import sys
from copy import deepcopy
from uhls.frontend import lower_source_to_uir
from uhls.interpreter import run_uir
from uhls.middleend.uir import format_module, verify_module
from uhls.middleend.passes.opt import ConstPropPass, CopyPropPass, DCEPass, SimplifyCFGPass
from uhls.middleend.passes.util import PassContext, PassManager

assert sys.platform == "emscripten", sys.platform
print("platform =", sys.platform)

source = """
int32_t main(void) {
    int32_t y = 7 + 1;
    return y * 2;
}
"""
module = lower_source_to_uir(source)
verify_module(module)
assert run_uir(module.get_function("main"), module=module).return_value == 16
print("Before:\\n" + format_module(module))
optimized = PassManager([
    ConstPropPass(), CopyPropPass(), DCEPass(), SimplifyCFGPass()
]).run(deepcopy(module), PassContext())
verify_module(optimized)
assert run_uir(optimized.get_function("main"), module=optimized).return_value == 16
print("After:\\n" + format_module(optimized))
print("PASS: both return 16")
''')]}


def prepare(repo: Path, workspace: Path, helper: Path, text_only: bool = False) -> str:
    src = repo / "src" / "uhls"
    notebook = repo / "doc" / "notebooks" / "uir_cookbook.ipynb"
    if not src.is_dir() or not notebook.is_file():
        raise ValueError("Run against a µhLS checkout containing src/uhls and doc/notebooks/uir_cookbook.ipynb")
    if workspace.is_symlink():
        raise ValueError("Refusing to replace a symlink workspace")
    reqs = dependencies(repo)
    version = fingerprint(src, reqs)
    original = json.loads(notebook.read_text(encoding="utf-8"))
    adapted = adapt_notebook(original, version, helper.read_text(encoding="utf-8"), text_only)
    if workspace.exists():
        if not (workspace / MARKER).is_file():
            raise ValueError(f"Refusing to replace unmarked directory {workspace}")
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    (workspace / MARKER).write_text("Generated files only.\n", encoding="utf-8")
    package = workspace / "package"
    shutil.copytree(src, package / "src" / "uhls",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    if (repo / "LICENSE").is_file():
        shutil.copy2(repo / "LICENSE", package / "LICENSE")
    (package / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=68", "wheel"]\nbuild-backend = "setuptools.build_meta"\n\n'
        f'[project]\nname = "{DISTRIBUTION}"\nversion = "{version}"\n'
        f'dependencies = {json.dumps(reqs)}\n\n'
        '[tool.setuptools]\ninclude-package-data = true\n\n'
        '[tool.setuptools.packages.find]\nwhere = ["src"]\ninclude = ["uhls", "uhls.*"]\n'
        'namespaces = true\n', encoding="utf-8")
    (package / "MANIFEST.in").write_text('graft src/uhls\nglobal-exclude *.py[cod]\n', encoding="utf-8")
    content = workspace / "lite" / "content"
    content.mkdir(parents=True)
    (workspace / "lite" / "pypi").mkdir()
    for name, nb in [("uir_cookbook.ipynb", adapted), ("00_smoke_test.ipynb", smoke_notebook(version))]:
        (content / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (workspace / "version.txt").write_text(version + "\n", encoding="utf-8")
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--repo", type=Path, default=here.parent.parent)
    parser.add_argument("--text-only", action="store_true", help="Show DOT text instead of browser Graphviz widgets")
    args = parser.parse_args()
    try:
        version = prepare(args.repo.resolve(), here / "_build", here / "graphviz_widget.py", args.text_only)
    except (OSError, ValueError, SyntaxError, KeyError) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Prepared {DISTRIBUTION} {version}; original sources and cookbook unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
