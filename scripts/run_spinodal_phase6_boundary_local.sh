#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 src/spinodal_phase6_boundary.py \
  --R-list "${PHASE6_R_LIST:-12}" \
  --N-map "${PHASE6_N_MAP:-6:512,12:1024,24:2048,48:4096,96:8192}" \
  --deltas "${PHASE6_DELTAS:-1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5}" \
  --epsilon-fractions "${PHASE6_EPSILONS:-0.025,0.05,0.10}" \
  --boundary-types "${PHASE6_BOUNDARIES:-ghost_dirichlet,open_fixed_denominator,open_renormalized}" \
  --right-boundary reflecting \
  --steady-tol "${PHASE6_STEADY_TOL:-1e-12}" \
  --max-steps "${PHASE6_MAX_STEPS:-20000}" \
  --minimum-tau-multiplier 10 \
  --fit-x-min-factor 2 \
  --primary-delta-max 3e-4 \
  --figures \
  --output-dir results/runs/poster_ABCD/phase6_boundary
