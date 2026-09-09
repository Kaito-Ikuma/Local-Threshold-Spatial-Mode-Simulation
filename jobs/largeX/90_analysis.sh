#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LARGEX_PY="${PHASE5_PY:-.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/phase5-largeX-matplotlib-${USER}}"
mkdir -p "$MPLCONFIGDIR"
INPUT="${LARGEX_INPUT:-results/runs/phase5_largeX/R024/production/primary}"
: "${LARGEX_DELTA_PS:?Set the operational delta_ps(Tobs=50) matching INPUT}"
"$LARGEX_PY" src/spinodal_phase5_largex_analysis.py analyze \
  --input-dir "$INPUT" --output-dir "$INPUT/largeX_analysis" \
  --delta-ps "$LARGEX_DELTA_PS" \
  --bootstrap-replicates "${LARGEX_BOOTSTRAP_REPLICATES:-5000}"
