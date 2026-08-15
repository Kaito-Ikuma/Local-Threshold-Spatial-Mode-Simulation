#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname -- "$SCRIPT_DIR")
PYTHON_BIN=${PYTHON_BIN:-python3}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" "$PROJECT_ROOT/src/spinodal_phase34.py" \
  --phase0-dir "$PROJECT_ROOT/results/runs/phase0_B2_R12" \
  --phase12-dir "$PROJECT_ROOT/results/runs/phase12_B2_R12" \
  --primary-delta-max 3e-4 \
  --qR-max-collapse 0.35 \
  --output-dir "$PROJECT_ROOT/results/runs/phase34_B2_R12"
