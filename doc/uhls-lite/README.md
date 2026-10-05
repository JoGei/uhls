# µhLS JupyterLite browser demo

The demo lives under `doc/uhls-lite/`. Its scripts find the repository two levels
above this directory, independently of the shell working directory. The generated
site has a project landing page at `/` and the JupyterLite application at `/lab/`.
There is no second root-level `uhls-lite/` tree to keep in sync.

No compiler source, original notebook, or root packaging metadata is changed.
Generated files and the suggested virtual environment stay inside
`doc/uhls-lite/`, ignored by the included `.gitignore`.

## Test locally

No GitHub Pages configuration, commit, push, or active workflow is needed.
From your µhLS repository root (the validated build used Python 3.12.3):

```bash
python3 -m venv doc/uhls-lite/.venv
source doc/uhls-lite/.venv/bin/activate
python -m pip install -r doc/uhls-lite/requirements.txt
bash doc/uhls-lite/build.sh
python -m http.server 8000 --bind 127.0.0.1 --directory doc/uhls-lite/_build/site
```

Open `http://localhost:8000/`, not the HTML file directly. The landing page links
to the available frontend lab and indexes the planned midend and backend labs;
JupyterLite is also available directly at `http://localhost:8000/lab/index.html`.
Keep the HTTP server running while testing.
Stop it with Ctrl+C when finished. The server only serves files: notebook Python
runs inside the browser via Pyodide. Internet is needed for initial installation
and the external runtime dependencies listed below; local testing is not the same
as a fully offline deployment.

Open `00_smoke_test.ipynb`, choose **Python (Pyodide)** if prompted, and run both
code cells. It asserts that `sys.platform == "emscripten"`; expected output includes
`platform = emscripten` and ends with `PASS: both return 16`. Then open
`uir_cookbook.ipynb` and use **Run → Run All Cells**.

For a static-site integrity check after building, run this while the virtual
environment is active. Running it from `_build/lite/` keeps JupyterLite's task
database out of the repository root.

```bash
(
  cd doc/uhls-lite/_build/lite
  jupyter lite check --lite-dir . --output-dir ../site
)
```

After changing compiler sources or the original cookbook, rebuild with
`bash doc/uhls-lite/build.sh`. Restart the browser kernel for a new compiler
revision. Browser-saved notebook copies can mask updated bundled notebooks;
export work you need before clearing site data or use a separate browser profile.
The generated browser notebook is a copy: edit the original cookbook for changes
that should be included in later builds.

## Troubleshooting

For compiler testing without graph widgets:

```bash
bash doc/uhls-lite/build.sh --text-only
```

This displays DOT as text. It is NOT an offline mode: Pyodide and Python packages
can still need downloads.

For missing imports, check the setup cell and selected kernel. For widget model
errors, ensure `anywidget` was installed in the build environment before building,
then rebuild and reload. The generated notebook requests the same anywidget
version as the build environment when available.

Native Verilator, Yosys, OpenROAD, shell commands, and native Graphviz executables
are outside the starter's scope. Use the existing compiler's Python APIs.

## Flow

1. Stages `src/uhls` and its package data under `_build/package/` and creates a
   pure-Python wheel named `uhls-browser-demo`. Its import package remains `uhls`.
   Sources and requirements contribute to a content-hashed version. Nothing is
   uploaded to PyPI.
2. Copies `doc/notebooks/uir_cookbook.ipynb` into the generated content. Its setup
   installs the bundled wheel through `piplite` instead of finding a Git checkout.
   A separate smoke-test notebook checks the compiler without graphs.
3. Adapts display-only `graphviz.Source` calls to a small anywidget/Viz.js helper.
   It does not implement Graphviz's `.render()` or subprocess APIs.
4. Builds the static site without JavaScript source maps in
   `doc/uhls-lite/_build/site/`, then installs the landing page at its root while
   retaining JupyterLite under `/lab/`.

The adapter checks the cookbook's setup markers and stops rather than silently
patching an incompatible notebook. The `_build/` directory is regenerated on
each build after checking its generated-files marker. Do not save work there.

## Documentation

- https://jupyterlite.readthedocs.io/en/stable/quickstart/standalone.html
- https://jupyterlite.readthedocs.io/en/stable/howto/pyodide/wheels.html
- https://jupyterlite.readthedocs.io/en/stable/howto/configure/advanced/offline.html
- https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows
- https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages
- https://docs.python.org/3/library/sys.html#sys.platform

This starter implements JupyterLite only. Its generated wheel can also be used
for a future marimo WebAssembly demo; the notebook loading and display integration
would need to be adapted and validated separately.
