#!/usr/bin/env bash
set -euo pipefail

source /eda-osic-tools/env.sh /eda-osic-tools/.env
if [[ -n "${VENV_ROOT:-}" && -f "${VENV_ROOT}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_ROOT}/bin/activate"
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt jupyter ipykernel graphviz
python3 -m ipykernel install --user --name uhls-devcontainer --display-name "Python (uhls devcontainer)"

marker_begin="# >>> uhls devcontainer >>>"
marker_end="# <<< uhls devcontainer <<<"
shell_block=$(cat <<'EOF'
# >>> uhls devcontainer >>>
source /eda-osic-tools/env.sh /eda-osic-tools/.env
if [ -n "${VENV_ROOT:-}" ] && [ -f "${VENV_ROOT}/bin/activate" ]; then
    source "${VENV_ROOT}/bin/activate"
fi
# <<< uhls devcontainer <<<
EOF
)

if [[ -f "${HOME}/.bashrc" ]] && ! grep -qF "${marker_begin}" "${HOME}/.bashrc"; then
    printf '\n%s\n' "${shell_block}" >> "${HOME}/.bashrc"
fi
