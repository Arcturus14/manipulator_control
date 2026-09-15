#!/usr/bin/env bash
set -euo pipefail

export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-/home/arclab-04/acados}"
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-test-manipulator}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
if [[ "${1:-}" == "gui" ]]; then
    shift
    exec python simulation_gui.py "$@"
fi
exec python main.py "$@"
