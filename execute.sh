#!/usr/bin/env bash
set -euo pipefail

export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-/home/arclab-04/acados}"
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-test-manipulator}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Prefer an explicitly selected interpreter, then an activated environment,
# and finally the repository's documented mppi Conda environment.  Non-login
# shells used by the GUI launcher do not necessarily expose a `python` command.
if [[ -n "${MANIPULATOR_PYTHON:-}" ]]; then
    PYTHON_BIN="${MANIPULATOR_PYTHON}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x "/home/arclab-04/miniconda3/envs/mppi/bin/python" ]]; then
    PYTHON_BIN="/home/arclab-04/miniconda3/envs/mppi/bin/python"
else
    echo "Error: Python interpreter not found." >&2
    echo "Activate the mppi environment or set MANIPULATOR_PYTHON." >&2
    exit 127
fi

if [[ "${1:-}" == "gui" ]]; then
    shift
    exec "${PYTHON_BIN}" simulation_gui.py "$@"
fi
exec "${PYTHON_BIN}" main.py "$@"
