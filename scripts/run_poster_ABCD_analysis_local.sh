#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 src/spinodal_poster_abcd.py \
  --micro-root results/runs/poster_ABCD/microscopic \
  --pseudospinodal-root results/runs/phase5_R_sweep \
  --phase6-dir results/runs/poster_ABCD/phase6_boundary \
  --R-list "${POSTER_R_LIST:-6,12,24,48,96}" \
  --B-R-list "${POSTER_B_R_LIST:-24,48}" \
  --epsilon-fraction 0.05 \
  --T-obs 50 \
  --qR-max 0.35 \
  --bootstrap-replicates "${POSTER_BOOTSTRAP_REPLICATES:-2000}" \
  --figures \
  --output-dir results/runs/poster_ABCD
