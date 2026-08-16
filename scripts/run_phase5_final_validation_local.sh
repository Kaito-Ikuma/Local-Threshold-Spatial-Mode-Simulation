#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/local_threshold_mpl}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" src/spinodal_phase5_final_validation.py analyze \
  --r-sweep-dir results/runs/phase5_R_sweep \
  --final-validation-dir results/runs/phase5_final_validation \
  --output-dir results/runs/phase5_final_validation

"$PYTHON_BIN" scripts/replot_phase5_final_validation.py \
  --input-dir results/runs/phase5_final_validation
