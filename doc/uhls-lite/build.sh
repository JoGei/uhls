#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
python "$HERE/prepare.py" --repo "$ROOT" "$@"
BUILD="$HERE/_build"
python -m build --wheel --no-isolation --outdir "$BUILD/lite/pypi" "$BUILD/package"
WHEELS=("$BUILD/lite/pypi/"*.whl)
if [[ ${#WHEELS[@]} -ne 1 || ! -f "${WHEELS[0]}" ]]; then
  echo "Expected one generated µhLS wheel" >&2
  exit 1
fi
(
  cd "$BUILD/lite"
  jupyter lite build \
    --lite-dir . \
    --contents content \
    --no-sourcemaps \
    --output-dir ../site
)
if [[ -f "$BUILD/site/lab/favicon.ico" ]]; then
  cp "$BUILD/site/lab/favicon.ico" "$BUILD/site/favicon.ico"
fi
cp "$HERE/landing/index.html" "$BUILD/site/index.html"
cp "$HERE/landing/landing.css" "$BUILD/site/landing.css"
cp "$ROOT/doc/figs/uhls_logo.svg" "$BUILD/site/uhls-logo.svg"
echo
echo "Serve locally with:"
echo "python -m http.server 8000 --bind 127.0.0.1 --directory '$BUILD/site'"
echo "Then open http://localhost:8000/"
