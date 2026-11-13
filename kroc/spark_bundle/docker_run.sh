#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_name="${KROC_SPARK_IMAGE:-kroc-mppi-spark:local}"
if [[ "${1:-}" == "build" ]]; then
    docker build --platform linux/arm64 \
        --build-arg "NGC_IMAGE=${KROC_NGC_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}" \
        -t "$image_name" "$bundle_dir"
    exit 0
fi
if [[ $# -eq 0 ]]; then
    echo 'Usage: bash docker_run.sh build | check | smoke | micro | validate | closedloop | all'
    exit 2
fi
docker run --rm --platform linux/arm64 --gpus all --shm-size=2g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$bundle_dir,target=/bundle" \
    -e MPLBACKEND=Agg -e MPLCONFIGDIR=/bundle/runtime/matplotlib \
    -e XDG_CACHE_HOME=/bundle/runtime/cache \
    "$image_name" python3 /bundle/run.py "$@" --require-arm64
