#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-/home/arclab-04/acados}"
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-test-manipulator}"

PYTHON_BIN="/home/arclab-04/miniconda3/envs/manipulator/bin/python"
CONFIG_PATH="${SCRIPT_DIR}/config.txt"
cd "${SCRIPT_DIR}"

EXECUTION_MODE="$("${PYTHON_BIN}" -m utils.execution_config \
    "${CONFIG_PATH}" --field mode)"

case "${EXECUTION_MODE}" in
    single)
        exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run.py" --config "${CONFIG_PATH}"
        ;;
    comparison)
        exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_comparison.py" --config "${CONFIG_PATH}"
        ;;
    *)
        echo "Unsupported execution mode: ${EXECUTION_MODE}" >&2
        exit 2
        ;;
esac
