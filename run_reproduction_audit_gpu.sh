#!/usr/bin/env bash
# Run in a terminal authenticated for the workload pool. Credentials are unchanged.
set -euo pipefail
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_PYTHON="${REPRO_PYTHON:-/mnt/shared-storage-user/xiejiaqing/miniconda3/envs/azr/bin/python}"
REPRO_CPUS="${REPRO_CPUS:-8}"
rlaunch \
  --cpu="${REPRO_CPUS}" \
  --gpu=1 \
  --charged-group="${REPRO_CHARGED_GROUP:-chemagent_gpu_pool}" \
  --memory=80000 \
  --private-machine=yes \
  --mount=gpfs://gpfs1/xiejiaqing:/mnt/shared-storage-user/xiejiaqing \
  -- "${REPRO_PYTHON}" "${REPRO_ROOT}/src/reproduce_paper_checkpoints.py" \
  --device cuda --threads "${REPRO_CPUS}" "$@"
