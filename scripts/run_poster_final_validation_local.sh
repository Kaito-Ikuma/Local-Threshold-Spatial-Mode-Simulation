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

"$PYTHON_BIN" src/spinodal_poster_final_validation.py \
  --output-root results/runs/poster_ABCD \
  --micro-modes results/runs/poster_ABCD/resultB_micro_mode_results.csv \
  --refinement-root results/runs/poster_ABCD/B_refinement \
  --pseudospinodal-root results/runs/phase5_R_sweep \
  --phase6-dir results/runs/poster_ABCD/phase6_boundary \
  --phase12-dir results/runs/phase12_B2_R12 \
  --q0-numeric-master results/runs/poster_ABCD/q0_full_numeric_validation/q0_numeric_all_5R.csv \
  --fully-numeric-only \
  --epsilon-fraction 0.05 \
  --T-obs 50 \
  --qR-max 0.35 \
  --primary-delta-max 3e-4
