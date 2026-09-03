#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/local_threshold_mpl}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" src/spinodal_R_sweep_analysis.py analyze \
  --gaussian-dir results/runs/gaussian_R_sweep \
  --micro-root results/runs/phase5_R_sweep \
  --R-list 6,12,24,48 \
  --output-dir results/runs/R_sweep_combined
