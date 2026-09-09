#!/bin/bash
set -euo pipefail
: "${LARGEX_DELTA_G:?Set one delta_G selected from the q0 scout result}"
: "${LARGEX_EPSILON:?Set LARGEX_EPSILON from the epsilon scout result}"
: "${LARGEX_FIT_END:?Set and freeze LARGEX_FIT_END from the q0 tau/survival rule}"
: "${LARGEX_RANKS:?Select the fastest safe rank count from the DBG benchmark}"
LARGEX_OUTPUT_TAG="${LARGEX_OUTPUT_TAG:-primary}"
case "$LARGEX_OUTPUT_TAG" in *[!A-Za-z0-9_.-]*) echo "Invalid LARGEX_OUTPUT_TAG" >&2; exit 2;; esac

mpirun ${NQSV_MPIOPTS:-} -np "$LARGEX_RANKS" \
  "$PHASE5_PY" src/spinodal_phase5_mpi.py \
  --analytic-references --B 2 --R 24 --N 4096 \
  --deltas "$LARGEX_DELTA_G" --auto-q-grid \
  --calibration-qR-max 0.15 --validation-qR-max 0.35 \
  --epsilon-fraction "$LARGEX_EPSILON" --epsilon-fractions "$LARGEX_EPSILON" \
  --M-total "$LARGEX_M" --block-size 32 \
  --kernel aggregated_exact --initialization prepared_metastable \
  --T-fixed 50 --fit-start 0 --fit-end "$LARGEX_FIT_END" \
  --survivor-cohort-end "$LARGEX_FIT_END" \
  --track-survival --rng-coupling-mode common_modes --harmonic-orders 3,5 \
  --epsilon-linearity-validated \
  --survival-criterion-primary 0.90 --survival-criterion-sensitivity 0.80 \
  --fit-protocol q0_tau_survival_frozen_v1 \
  --campaign-version 2026.09.09-phase5-largeX-v1 \
  --M-convergence-candidates 8192,16384,32768,65536 \
  --bootstrap-replicates 1000 --stage production --resume --no-figures \
  --max-runtime-seconds 53400 --task-id-prefix largeX_R024_production_ \
  --output-dir "results/runs/phase5_largeX/R024/production/$LARGEX_OUTPUT_TAG"
