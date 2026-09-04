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

OUTPUT_ROOT="results/runs/poster_ABCD/q0_full_numeric_validation"
PHASE6_DIR="results/runs/poster_ABCD/phase6_boundary"

# The Python driver reads N from the actual Phase6 CSV/summary and rejects a
# mismatch.  Each iteration contains exactly four deltas and mode_index=0.
for R in 6 24 48 96; do
  "$PYTHON_BIN" src/spinodal_q0_full_numeric_validation.py \
    --R-list "$R" \
    --no-aggregate \
    --output-root "$OUTPUT_ROOT" \
    --phase6-dir "$PHASE6_DIR" \
    --phase0-search-root results/runs \
    --existing-R12-dir results/runs/phase12_B2_R12
done

"$PYTHON_BIN" src/spinodal_q0_full_numeric_validation.py \
  --aggregate-only \
  --output-root "$OUTPUT_ROOT" \
  --phase6-dir "$PHASE6_DIR" \
  --phase0-search-root results/runs \
  --existing-R12-dir results/runs/phase12_B2_R12

# Join the five-R numerical master to the already measured Phase6 xi_bnd and
# regenerate the dedicated fully_numeric_z tables and figures.
scripts/run_poster_final_validation_local.sh
