#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/local_threshold_mpl}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" src/spinodal_gaussian_R_sweep.py \
  --R-list 6,12,24,48 \
  --reference-R 12 \
  --reference-N 1024 \
  --B 2.0 \
  --sigma-J 1.0 \
  --sigma-phi 0.06 \
  --delta-list 1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5 \
  --modes 0,1,2,3,4,5,6 \
  --output-dir results/runs/gaussian_R_sweep
